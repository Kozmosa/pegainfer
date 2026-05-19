use anyhow::{Result, ensure};
use half::bf16;
use pegainfer_core::tensor::{DeviceContext, HiddenStates};

use crate::{Config, device::activate};

#[derive(Default)]
pub(crate) struct DecodeCache {
    pub(crate) layers: Vec<LayerCache>,
}

#[derive(Default)]
pub(crate) struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerCache {
    pub(crate) fn len(&self, config: &Config) -> usize {
        let per_token = config.num_attention_heads * config.query_head_dim();
        if per_token == 0 {
            return 0;
        }
        self.keys.len() / per_token
    }
}

impl DecodeCache {
    pub(crate) fn new(config: &Config) -> Self {
        Self {
            layers: (0..config.num_hidden_layers)
                .map(|_| LayerCache::default())
                .collect(),
        }
    }
}

pub(crate) fn normalize_compressed_kv(
    config: &Config,
    kv_a_host: &[f32],
    norm_weight: &[f32],
    seq_len: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; config.kv_lora_rank * seq_len];
    for token in 0..seq_len {
        let src = &kv_a_host[token * config.kv_a_proj_rows()
            ..token * config.kv_a_proj_rows() + config.kv_lora_rank];
        let dst = &mut out[token * config.kv_lora_rank..(token + 1) * config.kv_lora_rank];
        rms_norm_host(src, norm_weight, config.rms_norm_eps, dst);
    }
    out
}

fn rms_norm_host(input: &[f32], weight: &[f32], eps: f32, out: &mut [bf16]) {
    let sum_sq = input.iter().map(|value| value * value).sum::<f32>();
    let inv_rms = (sum_sq / input.len() as f32 + eps).sqrt().recip();
    for ((dst, value), scale) in out.iter_mut().zip(input).zip(weight) {
        *dst = bf16::from_f32(value * inv_rms * scale);
    }
}

pub(crate) fn append_kv_and_build_queries(
    config: &Config,
    q_host: &[f32],
    kv_a_host: &[f32],
    kv_b_host: &[f32],
    start_pos: usize,
    seq_len: usize,
    queries: &mut [f32],
    cache: &mut LayerCache,
) {
    let num_heads = config.num_attention_heads;
    let q_head_dim = config.query_head_dim();
    let kv_b_stride = config.qk_nope_head_dim + config.v_head_dim;
    let mut key = vec![0.0f32; q_head_dim];

    for token in 0..seq_len {
        let pos = start_pos + token;
        let k_pe_raw = &kv_a_host[token * config.kv_a_proj_rows() + config.kv_lora_rank
            ..token * config.kv_a_proj_rows() + config.kv_lora_rank + config.qk_rope_head_dim];
        let k_pe = apply_rope(k_pe_raw, pos, config.qk_rope_head_dim, config.rope_theta);

        for head in 0..num_heads {
            let q_base = token * config.q_proj_rows() + head * q_head_dim;
            let query_base = (token * num_heads + head) * q_head_dim;
            queries[query_base..query_base + config.qk_nope_head_dim]
                .copy_from_slice(&q_host[q_base..q_base + config.qk_nope_head_dim]);
            let q_pe = apply_rope(
                &q_host[q_base + config.qk_nope_head_dim..q_base + q_head_dim],
                pos,
                config.qk_rope_head_dim,
                config.rope_theta,
            );
            queries[query_base + config.qk_nope_head_dim..query_base + q_head_dim]
                .copy_from_slice(&q_pe);

            let kv_b_base = token * config.kv_b_proj_rows() + head * kv_b_stride;
            key[..config.qk_nope_head_dim]
                .copy_from_slice(&kv_b_host[kv_b_base..kv_b_base + config.qk_nope_head_dim]);
            key[config.qk_nope_head_dim..q_head_dim].copy_from_slice(&k_pe);
            cache.keys.extend_from_slice(&key);
            cache.values.extend_from_slice(
                &kv_b_host[kv_b_base + config.qk_nope_head_dim
                    ..kv_b_base + config.qk_nope_head_dim + config.v_head_dim],
            );
        }
    }
}

fn apply_rope(input: &[f32], pos: usize, dim: usize, theta: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let inv_freq = 1.0f32 / theta.powf((2 * i) as f32 / dim as f32);
        let angle = pos as f32 * inv_freq;
        let cos = angle.cos();
        let sin = angle.sin();
        let x1 = input[i];
        let x2 = input[i + half];
        out[i] = x1 * cos - x2 * sin;
        out[i + half] = x2 * cos + x1 * sin;
    }
    out
}

pub(crate) fn compute_attention_host(
    config: &Config,
    queries: &[f32],
    cache: &LayerCache,
    start_pos: usize,
    seq_len: usize,
) -> Vec<f32> {
    let num_heads = config.num_attention_heads;
    let q_head_dim = config.query_head_dim();
    let value_dim = config.v_head_dim;
    let scale = (q_head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; seq_len * config.o_proj_cols()];

    for token in 0..seq_len {
        let kv_len = start_pos + token + 1;
        for head in 0..num_heads {
            let q_base = (token * num_heads + head) * q_head_dim;
            let query = &queries[q_base..q_base + q_head_dim];
            let mut scores = vec![0.0f32; kv_len];
            for (pos, score) in scores.iter_mut().enumerate() {
                let k_base = (pos * num_heads + head) * q_head_dim;
                let key = &cache.keys[k_base..k_base + q_head_dim];
                *score = dot(query, key) * scale;
            }
            let probs = softmax(&scores);
            let out_base = token * config.o_proj_cols() + head * value_dim;
            for (pos, prob) in probs.iter().enumerate() {
                let v_base = (pos * num_heads + head) * value_dim;
                let value = &cache.values[v_base..v_base + value_dim];
                for dim in 0..value_dim {
                    out[out_base + dim] += prob * value[dim];
                }
            }
        }
    }

    out
}

pub(crate) fn topk_softmax_routes(
    config: &Config,
    logits: &[f32],
    seq_len: usize,
) -> Vec<Vec<(usize, f32)>> {
    let mut routes = Vec::with_capacity(seq_len);
    for token in 0..seq_len {
        let scores =
            &logits[token * config.n_routed_experts..(token + 1) * config.n_routed_experts];
        let probs = softmax(scores);
        let mut indexed: Vec<_> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|(lhs_idx, lhs), (rhs_idx, rhs)| {
            rhs.partial_cmp(lhs)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| lhs_idx.cmp(rhs_idx))
        });
        indexed.truncate(config.num_experts_per_token);
        routes.push(indexed);
    }
    routes
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp = Vec::with_capacity(scores.len());
    let mut sum = 0.0f32;
    for score in scores {
        let value = (score - max).exp();
        exp.push(value);
        sum += value;
    }
    if sum == 0.0 {
        return vec![0.0; scores.len()];
    }
    exp.into_iter().map(|value| value / sum).collect()
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

pub(crate) fn hidden_to_bf16(ctx: &DeviceContext, hidden: &HiddenStates) -> Result<Vec<bf16>> {
    activate(ctx)?;
    let host = ctx.stream.clone_dtoh(&hidden.data)?;
    ctx.sync()?;
    Ok(host)
}

pub(crate) fn hidden_to_f32(ctx: &DeviceContext, hidden: &HiddenStates) -> Result<Vec<f32>> {
    Ok(hidden_to_bf16(ctx, hidden)?
        .iter()
        .map(|value| value.to_f32())
        .collect())
}

pub(crate) fn hidden_from_bf16_host(
    ctx: &DeviceContext,
    data: &[bf16],
    hidden_dim: usize,
    seq_len: usize,
) -> Result<HiddenStates> {
    activate(ctx)?;
    ensure!(
        data.len() == hidden_dim * seq_len,
        "hidden host data len mismatch: got {}, expected {}",
        data.len(),
        hidden_dim * seq_len
    );
    Ok(HiddenStates {
        data: ctx.stream.clone_htod(data)?,
        hidden_dim,
        seq_len,
    })
}

pub(crate) fn hidden_from_f32_host(
    ctx: &DeviceContext,
    data: &[f32],
    hidden_dim: usize,
    seq_len: usize,
) -> Result<HiddenStates> {
    let bf16_data: Vec<_> = data.iter().copied().map(bf16::from_f32).collect();
    hidden_from_bf16_host(ctx, &bf16_data, hidden_dim, seq_len)
}
