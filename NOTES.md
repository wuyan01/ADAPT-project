# ADAPT demo: inference speed + shadow patch

## Validated inference follow-up (2026-11-04)

The manifest now points to regenerated FP32 graphs with external noise inputs, ORT basic
constant folding, and 96 `SkipLayerNormalization` fusions per policy. Text has 1,865 nodes
(originally 5,801); goal has 1,866 (originally 5,770). The six new part files replace the
original six files. Full SHA-256 hashes and byte counts in the manifest were checked.

In an isolated headless Chrome session on this machine, 30 live-simulation state samples
gave text-policy median latency of 18.9 ms on the original WebGPU graph, 18.4 ms after
simplification alone, and 14.1 ms after fusion. The final CPU median was 8.8 ms.
The final goal-policy medians were 8.9 ms on CPU and 14.3 ms on WebGPU. These
are local smoke measurements including upload and action readback, not a portable
hardware benchmark; GPU remains slower than CPU here.

Graph capture requires stable GPU-backed input and output tensors in bundled ORT 1.23.2.
`policy-runner.mjs` supplies these and explicitly reads the actions back for MuJoCo.
A tiny Add graph passed real Chrome capture/replay with this runner, but the full policy
fails on its second run inside ORT with a `createBindGroup` undefined-buffer error.
The worker catches this during warm-up and recreates the ordinary WebGPU session.
`state.graph_capture` and the worker console report whether capture is actually active.
Do not attribute the measured improvement to capture.

ORT runtime optimization remains enabled: offline basic simplification does not replace
provider-specific runtime fusions. `onnxsim` crashed with the locally available packages,
so the script defaults to portable ORT basic passes; `--simplifier onnxsim` is optional.
Noise is externalized before simplification to permit deterministic equivalence checks.
The fusion reference is copied before optimization, and tolerance failures now abort
before writing assets. Against independently externalized original graphs, five identical
random feeds gave maximum action errors of 7.868e-6 (text) and 3.910e-5 (goal). This checks
numerical agreement, not identical trajectories to the original ORT random generator.

To regenerate from the original training exports, from `demo/`:

```sh
pip install onnx onnxruntime
python tools/optimize_policy.py /path/to/text_policy.onnx --name text --obs-dim 1088 --out assets --fuse
python tools/optimize_policy.py /path/to/goal_policy.onnx --name goal --obs-dim 994 --out assets --fuse
```

Merge the printed metadata (including `sha256`) into the corresponding manifest entries.
FP16 remains experimental; the worker currently loads the top-level FP32 assets.
Runner regressions: `node --test demo/tools/policy-runner.test.mjs` from the repository root.

The earlier patch notes below describe the initial proposal; the verified behavior above
supersedes its expectations about capture and disabling runtime optimization.

Drop the files in `demo/` over your `demo/` (they are patched copies of the current master; `coi-serviceworker.js`
and `tools/optimize_policy.py` are new). Everything in the browser code was syntax-checked but not run in a browser
here, so test in Chrome with the console open (`[ADAPT]` lines report backend, threads and graph capture).

## What was slow

`text` policy: 5,801 nodes / 30 M params / 369 matmuls. 1,746 `Constant`, 749 `Reshape`, 393 `Transpose`,
274 `Slice`, 200 `Shape`, 177 `Identity`, 130 `Cast`, 64 `Mod`, 32 `Trilu`, 4 `RandomNormalLike` (one per
denoising step). Critical path ≈ 1,050 nodes. At batch 1 that is launch/dispatch overhead, not compute.

| Cause | Effect | Fix |
|---|---|---|
| `RandomNormalLike` inside the graph, no WebGPU kernel | 4× GPU→CPU→GPU sync per inference; blocks graph capture; non-reproducible | `optimize_policy.py` turns them into `noise_k` inputs; worker fills them from its seeded RNG |
| Shape-dependent subgraphs (masks, reshapes) rebuilt every step despite fixed `[1,1088]` | hundreds of tiny CPU-side ops per step | onnxsim with frozen input shape (offline, once) |
| ORT optimizes the 5.8k-node graph in the browser on every load | slow load | ship the simplified graph, while retaining provider-specific runtime optimization |
| WebGPU: ~1k kernel launches per step | GPU slower than CPU | `enableGraphCapture: true` (works once all nodes are on WebGPU) |
| `ort.env.wasm.numThreads = 1`; GitHub Pages has no COOP/COEP | CPU path single-threaded | `coi-serviceworker.js` + `numThreads = min(4, cores-1)` when `crossOriginIsolated` |
| First GPU run compiles shaders | first seconds of a run stutter | 3 warm-up runs at session creation |
| Only inference was timed | can't tell physics from policy | new `physics_ms` metric |

Expected order of effect: offline simplification + noise externalization ≫ graph capture (GPU) / threads (CPU)
> warm-up. Run the script, then look at the console: if ORT logs that some nodes were not assigned to WebGPU,
the op list it names is what still needs to go before graph capture is accepted.

Optional extras in the script: `--fuse` (ORT contrib `MultiHeadAttention`/`SkipLayerNormalization`, all supported
by the WebGPU EP; the script checks outputs against the unfused model) and `--fp16` (halves the 120 MB download and
weight bandwidth on WebGPU; keep fp32 for the WASM path, where fp16 is slow). Use `--fuse` only after confirming
`--num-heads`/`--hidden` from the model config (hidden is 512 judging by the `[512, 1536]` QKV weights).

Not changed on purpose: the observation builder (≈1k floats of JS per step, negligible) and the serial
"physics waits for inference" loop, which is the demo's stated semantics.

## Shadow

`DirectionalLight.shadow.camera` was a fixed ±8 m box at the origin: outside it three.js returns "lit", inside it
the floor self-shadowed slightly (no bias), so the two regions looked different and the robot lost its shadow after
~8 m. `view.js` now moves the light and its target with the robot every pose update (snapped to shadow texels to
avoid swimming edges), sets `bias`/`normalBias`, tightens near/far, and uses `PCFSoftShadowMap`.
