# Kimi-K2 TP1 DP8 EP8 performance

> TL;DR: This ledger tracks pegainfer TP1+DP8+EP8 on 8x H20 against the vLLM TP1+DP8+EP8 bs64 target. The vLLM sustained bs64 `~106ms` TPOT is now explained by a DPLB/CUDA-graph bucket cliff: an uneven DP distribution such as `9,8,8,8,8,8,8,7` pads every rank from graph bucket 8 to 16 and doubles TPOT. Every pegainfer optimization must start from a profile, state the expected gain, show a microbench or isolated measurement, then pass correctness and service-level gates before commit.
>
> Last touched: 2026-05-25

## Target

| Item | Target |
| --- | ---: |
| Hardware | h20-100, 8x NVIDIA H20 |
| Model | `/data/models/Kimi-K2.5` |
| Shape | TP1 DP8 EP8 |
| Workload | prompt_len=1, output_len=128, max_concurrency=64, num_prompts=256 |
| vLLM baseline | output `594.57 tok/s`, TTFT p50/p99 `161.30/303.20ms`, TPOT p50/p99 `107.20/109.20ms`, ITL p50 `108.92ms` |
| Gate | `256/256` success, TPOT p50 `< 107.20ms`, TPOT p99 `< 109.20ms`, output `> 594.57 tok/s` |

Baseline source: h20-100 rerun with explicit bs64 warmup on 2026-05-25:
`/tmp/kimi-vllm-dp8-warmup-20260525/measure_bs64_o128_after_warmup.json`.
The older sweep in `docs/models/kimi-k2/vllm-h20-baseline.md` recorded bs64 TPOT p50/p99
`109.00/109.76ms`; the warmup-after rerun is slightly faster but still the same
100ms-class H20 shape.

The gate above is the sustained `num_prompts=256, max_concurrency=64` client shape,
not a one-shot 64-request pure-decode wave. A separate command audit on 2026-05-25
showed that vLLM can report `~50ms` TPOT for a single 64-request wave, then return to
`~106ms` TPOT when the benchmark continuously refills another 192 requests. Treat these
as different workloads.

## Method

Performance work in this file follows this loop:

1. Profile: record the service JSON/log, in-process JSON, and nsys sqlite/tail report when profiling is needed.
2. Motivation and expected gain: name the bottleneck and estimate the target metric movement.
3. Microbench: isolate the changed stage, or explain why the service/in-process measurement is the smallest meaningful unit.
4. Correctness: keep generated-token hash distributions, mismatch counts, and any relaxed tolerance rationale.
5. Decision: keep, reject, defer, or revert; every kept optimization gets a commit.

For TP1 DP8, correctness checks must include uneven per-rank active rows and empty-rank EP participation, because PPLX collectives still require all ranks to enter each MoE layer in the same order.

## Unified Commands

Build on h20-100:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
/root/.cargo/bin/cargo build --release -p pegainfer-server \
  --features kimi-k2-pplx-ep --bin pegainfer --bin bench_serving
```

In-process bs64:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp1dp8 \
target/release/bench_serving \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi-tp1dp8/tp1dp8_bs64_o128_${COMMIT}.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1
```

Service bs64, same client shape as vLLM:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp1dp8 \
target/release/pegainfer --model-path /data/models/Kimi-K2.5 --port 8124 --cuda-graph false
```

```bash
source /root/develop/xingming/vllm_test/.venv/bin/activate
vllm bench serve \
  --backend openai \
  --model /data/models/Kimi-K2.5 \
  --tokenizer /data/models/Kimi-K2.5 \
  --trust-remote-code \
  --base-url http://127.0.0.1:8124 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 256 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/kimi-tp1dp8-service \
  --result-filename pegainfer_tp1dp8_bs64_${COMMIT}.json
```

vLLM TP1 DP8 EP8 baseline server:

```bash
cd /root/develop/xingming/vllm_test
source .venv/bin/activate
vllm serve /data/models/Kimi-K2.5 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --api-server-count 1 \
  --served-model-name kimi-k2.5 \
  --port 8123 \
  --max-num-seqs 64 \
  --max-model-len 4096
```

Use the served model name on the client. vLLM 0.19.0 returns 404 for
`--model /data/models/Kimi-K2.5` in the single-API-server setup.

```bash
cd /root/develop/xingming/vllm_test
source .venv/bin/activate
vllm bench serve \
  --backend openai \
  --model kimi-k2.5 \
  --tokenizer /data/models/Kimi-K2.5 \
  --trust-remote-code \
  --base-url http://127.0.0.1:8123 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 256 \
  --max-concurrency 64 \
  --request-rate inf \
  --ignore-eos \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/kimi-vllm-dp8-cmdcheck-20260525 \
  --result-filename api1_maxseq64_measure_bs64_o128_after_warmup_modelname.json
```

nsys profile:

```bash
cd /root/develop/xingming/pegainfer
mkdir -p /tmp/kimi-profile
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp1dp8 \
nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --cuda-graph-trace=node --export=sqlite \
  -o /tmp/kimi-profile/tp1dp8_bs64_o128_${COMMIT} \
  target/release/bench_serving \
    --model-path /data/models/Kimi-K2.5 \
    --cuda-graph false \
    --cuda-profiler-capture \
    --format json \
    --out /tmp/kimi-profile/tp1dp8_bs64_o128_${COMMIT}.json \
    request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 1 --iters 1

uv run --no-project python tools/nsys_tail_stats.py \
  /tmp/kimi-profile/tp1dp8_bs64_o128_${COMMIT}.sqlite \
  --out /tmp/kimi-profile/tp1dp8_bs64_o128_${COMMIT}_tail.md
```

## Optimization Log

### O1 - prompt_len=1 admission goes through decode

Status: keep. Baseline implementation: `8946078`. Safety follow-ups: `64192bb`, `0c23389`.

Profile:

- Code inspection showed TP1 DP8 uses `DpCoordinator`, not the TP8 `KimiK2Scheduler` prompt_len1 batch path.
- Old admission ran each prompt_len=1 request through `synchronized_prefill`, with `decode_batch_size=1`, and padding ranks doing dummy prefill. At bs64 that is 64 synchronized prefill waves.
- Old `MAX_BATCH_PER_DP=4` capped global active requests at 32, so bs64 could not occupy all requested rows.

Motivation and expected gain:

- prompt_len=1 is semantically a decode step at position 0: consume one token, append KV at position 0, produce the first generated token.
- Replace 64 serialized prompt prefill waves with one DP-wide decode admission wave.
- Raise per-DP slots to 8 so TP1 DP8 can hold the full bs64 workload.
- Expected gain: large TTFT reduction and service throughput improvement; TPOT should use rank-local bs8 instead of two bs32 waves.

Change:

- `pegainfer-kimi-k2/src/runner/engine.rs`
  - `MAX_BATCH_PER_DP: 4 -> 8`.
  - Added prompt_len1 admission batching in `DpCoordinator`.
  - For prompt_len1 requests, send `StepCommand::Decode { positions: vec![0], slots, decode_batch_size: MAX_BATCH_PER_DP }` instead of `Prefill`.
  - Empty ranks still run padding decode with the same arena capacity to preserve PPLX collective order.
  - Existing active rows are included in the same prompt_len1 admission decode command; padding rows can only use free slots.
  - Ordinary prefill padding ranks write the dummy token into a free slot, not fixed slot 0. If any rank lacks a safe padding slot, that request remains pending.

Correctness constraints:

- In TP1 DP8, `decode_batch_size` means decode arena capacity, not active row count. Keep it fixed at `MAX_BATCH_PER_DP` for decode, prompt_len1 admission, padding decode, and ordinary prefill.
- Slot IDs are decode arena row IDs. A request must keep the same arena bucket for prefill and all decode steps, otherwise its KV cache lives in a different arena.
- PPLX decode scratch capacity must be identical across ranks even when active row counts differ.
- Padding decode and padding prefill execute real kernels and can write KV. They may only target unoccupied slots.
- Every synchronized step must drain one result from every DP rank, including the error path, before the next command is sent.
- Padding prefill failures are request failures; the owner request must not become active unless every rank completed its synchronized prefill step.
- A missing rank forward thread is fatal for the process. Continuing with a partial DP command would leave surviving ranks inside unmatched PPLX collectives.
- prompt_len1 admission at `append_position=0` must install request state after the first token, or finish/error the request in the same result pass.

Microbench:

- Remote build passed on h20-100 at `0c23389`.
- Smoke command:

```bash
PEGAINFER_KIMI_PARALLEL=tp1dp8 target/release/bench_serving \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi-tp1dp8/tp1dp8_bs64_o5_64192bb_smoke.json \
  request --prompt-len 1 --output-len 5 --concurrency 64 --warmup 0 --iters 1
```

- Smoke result after stable-arena safety fix: `64/64` success,
  `steady_tpot_ms` p50/p95/p99 `37.21/37.41/37.42ms`, first decode step p50 `38.47ms`.

Correctness:

- Smoke generated all 5 tokens for every request without PPLX collective mismatch or slot-state failure.
- bs8/o8 deterministic smoke generated `8/8` full traces with one hash,
  `/tmp/kimi-tp1dp8/prompt1_decode_admission_bs8_o8_correctness.json`.
- Local coordinator tests cover sparse logical slots, prompt_len1 admission mixed with active rows,
  padding decode arena capacity, and ordinary prefill padding slot selection:

```bash
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
cargo test -r -p pegainfer-kimi-k2 --features pplx-ep runner::engine::tests --no-fail-fast
```

- Local result: `5 passed`.
- h20-100 result at `0c23389`: `5 passed`.
- Mixed-arrival service test, `/tmp/kimi-tp1dp8-service/pegainfer_tp1dp8_mixed_arrival_prompt1_o64_0c23389.json`:
  `64/64` success with `--request-rate 16`, peak concurrent requests `54`, TTFT p50/p99
  `58.10/110.88ms`, TPOT p50/p99 `35.91/37.63ms`. This covers prompt_len1
  admissions landing while existing decode slots are active.

Performance:

- In-process, `/tmp/kimi-tp1dp8/tp1dp8_bs64_o128_0c23389_w1_i1.json`:
  `64/64` success, TTFT p50/p99 `74.62/77.19ms`, first decode p50/p99
  `38.23/38.24ms`, steady TPOT p50/p95/p99 `40.10/43.32/43.72ms`.
- Service, same `vllm bench serve` client as vLLM,
  `/tmp/kimi-tp1dp8-service/pegainfer_tp1dp8_bs64_o128_0c23389_after_warmup.json`:
  `256/256` success, output `1336.35 tok/s`, TTFT p50/p99 `105.31/127.81ms`,
  TPOT p50/p95/p99 `47.34/47.70/47.71ms`, ITL p50/p99 `47.84/50.69ms`.
- vLLM warmup-after baseline,
  `/tmp/kimi-vllm-dp8-warmup-20260525/measure_bs64_o128_after_warmup.json`:
  `256/256` success, output `594.57 tok/s`, TTFT p50/p99 `161.30/303.20ms`,
  TPOT p50/p95/p99 `107.20/109.00/109.20ms`, ITL p50/p99 `108.92/116.35ms`.

vLLM baseline diagnosis, h20-100, vLLM `0.19.0`, NCCL/AgRs path:

- Startup sanity: `max_seq_len=4096` is confirmed in the log; active context is only
  about 129 tokens. Workers use `nccl==2.27.5` and `AgRsAll2AllManager`. `pplx` is
  removed/falls back in this vLLM, and DBO requires DeepEP backends, so neither is a
  valid NCCL baseline knob here.
- Command sanity: use `--api-server-count 1`, `--max-num-seqs 64`, and client
  `--model kimi-k2.5`. This removes API-process routing noise and lowers graph
  capture from largest `512` to `128`, but does not by itself fix sustained TPOT.
- Single-wave vs sustained: the same server reports `50.45/50.46ms` TPOT p50/p99
  for one 64-request wave, but `106.92/108.73ms` for sustained
  `num_prompts=256,max_concurrency=64`.

Pinned DP-rank controls explain the cliff:

| Run | DP-rank distribution | Global output | TPOT p50/p99 | Artifact |
| --- | --- | ---: | ---: | --- |
| balanced | `8,8,8,8,8,8,8,8` | `1192.22 tok/s` | `48.41/48.95ms` | `/tmp/kimi-vllm-dp8-dplb-20260525/balanced_8x8/` |
| one-rank over bucket | `9,8,8,8,8,8,8,7` | `640.94 tok/s` | `96.01/97.34ms` | `/tmp/kimi-vllm-dp8-dplb-20260525/skew_98888887/` |
| observed-like skew | `8,9,9,9,8,7,7,7` | `612.12 tok/s` | `99.80/99.99ms` | `/tmp/kimi-vllm-dp8-dplb-20260525/skew_89998777/` |

Mechanism:

- vLLM DPLB minimizes `waiting * 4 + running`
  (`vllm/v1/engine/core_client.py:1337-1360`) and refreshes local counts from
  coordinator stats (`core_client.py:1263-1274`). Sustained refill logs show small
  imbalances such as `8,9,9,9,8,7,7,7`, `11,7,7,7,7,8,9,8`, and
  `10,9,8,7,7,7,7,9`.
- CUDA Graph dispatch pads non-exact sizes to the next captured bucket
  (`vllm/v1/cudagraph_dispatcher.py:71-90,140-151`), so local batch `8` uses bucket
  8 and local batch `9` uses bucket 16.
- DP coordination pads every rank to the maximum padded size when CUDA Graph is
  active (`vllm/v1/worker/dp_utils.py:78-88,148-160`;
  `gpu_model_runner.py:3616-3637`). One rank at 9 therefore makes the whole DP group
  execute bucket 16.

Decision for vLLM interpretation: the surprising 2x TPOT is a DPLB plus DP CUDA
Graph padding cliff. Out-of-box sustained serving is correctly reported as
`~106ms`; balanced pinned capability at bs64 is `~48-50ms`. A vLLM-side fix needs
bucket-aware DP routing or explicit router/header assignment for controlled bs64
benchmarks.

Decision:

- Keep. O1 moves prompt_len=1 onto the correct decode shape and clears the current H20
  vLLM bs64 TPOT/output gate. Follow-up profiles should focus on lowering pegainfer service
  TPOT from `47ms` toward the H200-reported 30ms-class expectation if that target is confirmed
  on comparable hardware.

## Open Questions

- The H20 vLLM TP1 DP8 EP8 sustained-vs-balanced discrepancy is explained by the
  DPLB/CUDA-graph bucket cliff above. The remembered 30ms-class TPOT is still not
  reproduced on H20; it may have been measured on H200 or with a different vLLM
  build/version/runtime flag set.
- `vllm bench serve` can report `max_concurrent_requests=128` while the command uses
  `--max-concurrency 64`. Source inspection shows the client semaphore is real, but
  `max_concurrent_requests` is computed from one-second buckets and counts both
  requests ending and requests starting inside the same bucket. Treat this field as a
  coarse reporting artifact for refill-heavy runs; rely on the command shape, completed
  traces, throughput, and TPOT/ITL percentiles.
