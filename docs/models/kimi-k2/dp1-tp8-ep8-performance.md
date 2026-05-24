# Kimi-K2 DP1 TP8 EP8 Performance

> **TL;DR:** DP1 TP8 EP8 的性能主线从 correctness baseline
> `72c770b` 开始。目标是在 H20 ×8、bs64、decode-heavy 服务口径下超过
> vLLM `0.19.0` 的 bs64 baseline：output `583.9 tok/s`，TPOT median
> `109.00ms`。
>
> **Status:** Project doc opened. No performance optimization is accepted here
> until it has a correctness gate and its own commit.

## Target

| Item | Target / baseline |
| --- | --- |
| Machine | `h20-100`, 8× NVIDIA H20 |
| Model | `/data/models/Kimi-K2.5` |
| Shape | DP1 TP8 EP8 |
| Primary workload | `input_len=1`, `output_len=128`, `ignore-eos`, `bs=64` |
| vLLM baseline | TP1 DP8 EP8, `vllm bench serve`, output `583.9 tok/s`, TPOT median `109.00ms`, TPOT p99 `109.76ms` |
| PegaInfer goal | output tok/s `> 583.9` at bs64, while preserving token correctness |

The comparison target comes from [vllm-h20-baseline.md](vllm-h20-baseline.md).
The correctness ground truth starts from
[pplx-ep-correctness.md](pplx-ep-correctness.md): TP8 NCCL and TP8 PPLX both
produce 64-token hash `4920f088c2338236` for the baseline probe.

## Gate Rules

Every kept optimization needs all of these recorded before commit:

| Gate | Requirement |
| --- | --- |
| Profile | Start from an observed profile or benchmark delta. Record the command, output path, and the measured bottleneck or symptom. |
| Motivation / expected gain | State why the change should help and the expected size/direction of the win before implementing it. |
| Microbench | Add or run the smallest probe that isolates the changed subsystem when practical. If no microbench is practical, record why and use the closest lower-level measurement. |
| Correctness | Record the exact command, output file, token hash, and comparison target. For TP8/PPLX changes, compare against the TP8 NCCL baseline unless a stronger reference is documented. |
| Performance | Record bs64 service numbers and the lower-level in-process probe that explains the delta. |
| Scope | State whether the optimization targets frontend/scheduler, CUDA graph, collectives, MLA, MoE, or sampling. |
| Revert line | Record the measurable regression that would make the change revert-worthy. |
| Commit | Commit the code and this doc update together. |

No optimization is accepted on performance numbers alone.

Preferred entry shape:

```text
Profile:
  <command + report path + bottleneck>
Motivation / expected gain:
  <why this change should move bs64, and by roughly how much>
Microbench:
  <isolated probe, or the reason a subsystem-only probe is not available>
Correctness gate:
  <hash / trace / reference path>
Performance gate:
  <bs64 service number + supporting in-process/profile number>
Decision:
  <keep/reject/defer + commit>
```

This is a discipline, not a rigid template. The important part is that future
readers can reconstruct why an optimization was attempted, what it was expected
to buy, and which evidence made it worth keeping.

## Canonical Bs64 Pressure Test

Use this exact service pressure-test shape for bs64 comparisons. Do not change
prompt/output length, request count, request rate, concurrency, percentiles,
streaming mode, or `ignore-eos` when reporting numbers against the vLLM bs64
baseline.

Server:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep -- \
  --model-path /data/models/Kimi-K2.5 \
  --port 8124 \
  --cuda-graph true
```

Client:

```bash
cd /root/develop/xingming/pegainfer
COMMIT=$(git rev-parse --short HEAD)
mkdir -p /tmp/kimi-bs64-baseline
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
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-dir /tmp/kimi-bs64-baseline \
  --result-filename pegainfer_tp8_pplx_bs64_${COMMIT}.json \
  2>&1 | tee /tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_${COMMIT}.log
```

Required report fields:

| Field | Value |
| --- | --- |
| `--random-input-len` | `1` |
| `--random-output-len` | `128` |
| `--random-range-ratio` | `0` |
| `--num-prompts` | `256` |
| `--max-concurrency` | `64` |
| `--request-rate` | `inf` |
| `--ignore-eos` | enabled |
| `--percentile-metrics` | `ttft,tpot,itl` |
| `--metric-percentiles` | `50,95,99` |

## Correctness Probe

Run this before accepting a performance change:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_correctness64.json \
  request --output-len 64 --warmup 0 --iters 1
```

## Optimization Ledger

| ID | Date | Commit | Area | Change | Correctness gate | bs64 result | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 2026-05-25 | `72c770b` | correctness | TP8 PPLX baseline fixed; no performance claim | TP8 NCCL/PPLX 64-token hash `4920f088c2338236` | Not measured | Keep as ground truth |
| B1 | 2026-05-25 | `d639e55` code, `df1cd18` command doc | scheduler / service profile | Canonical bs64 pressure baseline before performance work | No code change after B0; PPLX correctness baseline remains `4920f088c2338236` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json`: output `137.51 tok/s`, TPOT p50/p95/p99 `26.40/28.13/28.46ms`, TTFT p50/p99 `54.76/58.68s`, 256/256 success | Keep as profile baseline; first optimization should address 4-row scheduling/admission before kernel work |
| O1 | 2026-05-25 | this commit | scheduler / decode arena | Raise DP1 TP8 admission to bs64; allocate decode arenas lazily in `1/2/4/8/16/32/64` buckets; preflight arena allocation on all TP ranks before prefill collectives | `/tmp/kimi_pplx_tp8_correctness64_o1_bucket.json`: TP8 PPLX 64-token hash `4920f088c2338236` | `/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.json`: output `145.18 tok/s`, TPOT p50/p95/p99 `195.07/221.08/224.72ms`, TTFT p50/p99 `31.00/35.76s`, 256/256 success | Keep as bs64 enabling baseline; not enough for vLLM target, next profile must attack bs64 kernel/communication cost |

### B1 Profile Notes

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json
```

Observed:

- bs64 output throughput is `137.51 tok/s`, far below vLLM bs64 `583.9 tok/s`.
- TPOT p50 is `26.40ms`, much lower than vLLM bs64 TPOT p50 `109.00ms`.
- TTFT p50 is `54.76s`, showing requests are queued in long waves.
- Current TP8 scheduler cap is still `KIMI_RUNNER_MAX_BATCH = 4`, so bs64 service
  pressure effectively runs as repeated 4-row decode waves.

Motivation / expected gain:

Raising the DP1 TP8 admission/arena path beyond 4 rows should attack the main
service-throughput gap directly. If per-token TPOT stayed near the B1 value,
bs64 throughput would have roughly 4x headroom before kernel scaling becomes the
dominant limit. The actual gain must be measured because MLA/MoE kernels,
collectives, scratch size, and graph capture may scale nonlinearly with batch.

Microbench:

B1 is a service profile, not an optimization. The next optimization must add a
lower-level probe for the changed layer, for example an in-process bs sweep or a
decode arena/scheduler trace that confirms active rows > 4 before rerunning the
canonical bs64 pressure command.

### O1 Lazy Bucketed Bs64 Decode Arenas

Profile:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_d639e55.json
```

Observed:

- Canonical bs64 service throughput was only `137.51 tok/s`, with TTFT p50
  `54.76s`.
- TPOT p50 was `26.40ms`, which was good for each 4-row wave but did not
  translate into bs64 service throughput.
- Code profile: TP8 scheduler admitted at most `KIMI_RUNNER_MAX_BATCH = 4`,
  and worker startup allocated all decode arenas eagerly up to the worker cap.

Motivation / expected gain:

Raising the scheduler and worker cap to 64 removes the obvious admission limit.
Decode arenas are allocated lazily in power-of-two buckets so canonical bs64
uses one bs64 KV/scratch/graph arena without allocating every size from 1 to 64.
The rank preflight makes allocation failure happen before prefill/decode
collectives, avoiding a partial-rank failure mode. Expected direction: much
lower bs64 TTFT and enough active rows to expose the real bs64 kernel and PPLX
communication cost.

Microbench:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph true \
  --format json \
  --out /tmp/kimi_pplx_tp8_o1_bucket_micro_bs64.json \
  request --prompt-len 1 --output-len 128 --concurrency 64 --warmup 0 --iters 1
```

Result:

- Output path: `/tmp/kimi_pplx_tp8_o1_bucket_micro_bs64.json`.
- Workload confirmed `concurrency=64`, `output_len=128`, all `64` traces had
  length `128`.
- In-process wall throughput, computed as `64 * 128 / max_e2e`, was about
  `226.9 tok/s` (`max_e2e=36.108s`).
- Steady TPOT p50/p95/p99 was `178.35/201.96/218.85ms`.

Correctness gate:

```bash
cd /root/develop/xingming/pegainfer
CUDA_HOME=/usr/local/cuda \
NVCC=/usr/local/cuda/bin/nvcc \
LD_LIBRARY_PATH=/tmp/pegainfer-nccl-lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
PEGAINFER_CUDA_SM=90a \
PEGAINFER_TRITON_PYTHON=/root/develop/xingming/pegainfer/.triton-venv/bin/python \
PEGAINFER_KIMI_PARALLEL=tp8dp1 \
cargo run --release -p pegainfer-server --features kimi-k2-pplx-ep --bin bench_serving -- \
  --model-path /data/models/Kimi-K2.5 \
  --cuda-graph false \
  --format json \
  --out /tmp/kimi_pplx_tp8_correctness64_o1_bucket.json \
  request --output-len 64 --warmup 0 --iters 1
```

Result: generated-token hash `4920f088c2338236`, matching the TP8 NCCL/PPLX
baseline.

Performance gate:

Canonical bs64 service result:

```text
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.log
/tmp/kimi-bs64-baseline/pegainfer_tp8_pplx_bs64_o1-bucket-07d6a40.json
```

Observed:

- Successful requests: `256/256`.
- Output throughput: `145.18 tok/s` vs B1 `137.51 tok/s`.
- Peak output throughput: `504.00 tok/s` vs B1 `168.00 tok/s`.
- TTFT p50/p95/p99: `31.00/35.23/35.76s` vs B1 p50/p99
  `54.76/58.68s`.
- TPOT p50/p95/p99: `195.07/221.08/224.72ms`.

Decision:

Keep. O1 preserves token correctness and turns bs64 into real 64-row decode
waves, but the service output throughput is still far below the vLLM `583.9
tok/s` target. The next accepted optimization needs a profile of the bs64
decode step itself, especially PPLX MoE routing/combine, MLA decode, and TP
collectives.

## Candidate Queue

| Priority | Area | Hypothesis | Correctness risk |
| --- | --- | --- | --- |
| P0 | bs64 profile | Profile one measured bs64 decode step after O1 to split PPLX MoE, MLA, TP collectives, and host replay overhead. | Low: profile-only, but capture must not change request shape. |
| P0 | PPLX / MoE | O1 shows real bs64 waves but TPOT p50 is `195.07ms`; routed expert dispatch/combine or Marlin route capacity may now dominate. | High: routed expert and combine weights are correctness-sensitive. |
| P0 | CUDA Graph | Reduce bs64 first-step graph capture/replay and metadata overhead after kernel profile identifies host or graph-node cost. | Medium: graph replay must preserve per-row metadata and PPLX participation. |
| P1 | frontend | Measure HTTP/streaming overhead separately from in-process TPOT. | Low for model math, medium for serving semantics. |
| P1 | collectives | Profile TP all-reduce and routed combine tail at bs64. | Medium: BF16/F32 collective boundary is correctness-sensitive. |
| P2 | MLA/MoE | Retune batch-shape kernels only after scheduler and graph bottlenecks are visible. | High: routed expert and MLA cache layout are easy to perturb. |

## Rejected / Deferred

| Date | Idea | Reason |
| --- | --- | --- |
| 2026-05-25 | Use TP1/DP8 correctness as the baseline for this doc | Deferred. TP1/DP8 matched short probes but diverged at 32 tokens, so DP1 TP8 work uses TP8 NCCL/PPLX baseline first. |
