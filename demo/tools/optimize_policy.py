#!/usr/bin/env python3
"""Shrink the ADAPT policy graphs for in-browser inference.

The exported policy.onnx is a 4-step denoising loop unrolled into ~5,800 nodes for only ~30M parameters.
At batch 1 inference time is dominated by per-node overhead (WASM dispatch, WebGPU kernel launches), not FLOPs.
This script:

  1. Freezes the input shape (batch 1) and constant-folds everything that depends only on shapes:
     Shape/Gather/Concat/Reshape chains, Trilu/Where causal masks, Constant/Identity/Cast nodes.
  2. Turns each in-graph RandomNormalLike into an explicit graph input (noise_0 … noise_3). The browser worker
     fills them from its seeded RNG, so runs become reproducible, and the WebGPU graph no longer has to round-trip
     to the CPU four times per step (RandomNormalLike has no WebGPU kernel). This also makes ORT graph capture
     possible, which removes most remaining launch overhead.
  3. Optionally fuses attention / layer-norm / gelu into ORT contrib ops (--fuse) and writes an fp16 copy
     for WebGPU (--fp16).
  4. Splits the result into <= 48 MiB parts and prints the manifest.json snippet for the policy entry.

Usage (from the demo/ directory, with the original policy.onnx from the training export):

  pip install onnx onnxruntime
  python tools/optimize_policy.py path/to/text_policy.onnx  --name text --obs-dim 1088 --out assets
  python tools/optimize_policy.py path/to/goal_policy.onnx  --name goal --obs-dim 994  --out assets
  # optional: --simplifier onnxsim (requires onnxsim), --fp16 (requires onnxconverter-common)

Then paste the printed snippet into manifest.json (parts, bytes, sha256, extra_inputs, preoptimized).
The worker still enables ORT's provider-specific graph optimizations after simplification.
"""
import argparse, hashlib, json, pathlib, sys, tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

PART_BYTES = 48 * 1024 * 1024  # matches the existing 50331648-byte parts


def simplify(model, obs_dim, method='ort'):
    if method == 'ort':
        import onnxruntime as ort
        # Basic passes are provider-independent; do not serialize CPU-specific
        # extended/layout fusions into a graph that also needs to run on WebGPU.
        with tempfile.TemporaryDirectory(prefix='adapt-optimize-') as directory:
            path = pathlib.Path(directory) / 'simplified.onnx'
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            options.optimized_model_filepath = str(path)
            options.intra_op_num_threads = 1
            ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
            return onnx.load(str(path))
    try:
        import onnxsim
    except ImportError:
        sys.exit('pip install onnxsim')
    model, ok = onnxsim.simplify(model, overwrite_input_shapes={'obs': [1, obs_dim]})
    if not ok:
        sys.exit('onnxsim reported a validation failure')
    return model


def externalize_noise(model):
    """Replace RandomNormalLike nodes with graph inputs. Returns {input_name: shape}."""
    model = shape_inference.infer_shapes(model)
    g = model.graph
    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        dims = [d.dim_value if d.HasField('dim_value') else None for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    for init in g.initializer:
        shapes[init.name] = list(init.dims)
    extra = {}
    keep = []
    k = 0
    for node in g.node:
        if node.op_type not in ('RandomNormalLike', 'RandomNormal'):
            keep.append(node)
            continue
        if node.op_type == 'RandomNormalLike':
            shape = shapes.get(node.input[0])
        else:
            shape = next(list(a.ints) for a in node.attribute if a.name == 'shape')
        if shape is None or any(d is None for d in shape):
            sys.exit(f'{node.name}: could not infer a static shape for the noise tensor ({shape}); run onnxsim first')
        for a in node.attribute:
            if a.name in ('mean', 'scale') and abs(a.f - (0.0 if a.name == 'mean' else 1.0)) > 1e-6:
                sys.exit(f'{node.name}: non-standard {a.name}={a.f}; teach the worker to apply it')
        name = f'noise_{k}'
        k += 1
        # Keep the node's output tensor name so consumers need no rewiring; the new input simply produces it.
        g.input.append(helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, shape))
        extra[node.output[0]] = [int(d) for d in shape]
        print(f'  {node.op_type} {node.name!r} -> graph input {node.output[0]!r} {shape}')
    del g.node[:]
    g.node.extend(keep)
    # Tidy input names: rename the promoted inputs to noise_k everywhere.
    renamed = {}
    for i, (old, shape) in enumerate(extra.items()):
        new = f'noise_{i}'
        renamed[new] = shape
        for vi in g.input:
            if vi.name == old:
                vi.name = new
        for node in g.node:
            for j, inp in enumerate(node.input):
                if inp == old:
                    node.input[j] = new
    return model, renamed


def fuse(model, num_heads, hidden):
    try:
        from onnxruntime.transformers.optimizer import optimize_by_fusion
        from onnxruntime.transformers.fusion_options import FusionOptions
    except ImportError:
        sys.exit('pip install onnxruntime')
    options = FusionOptions('bert')
    options.enable_skip_layer_norm = True
    options.enable_bias_skip_layer_norm = True
    optimizer = optimize_by_fusion(model, model_type='bert', num_heads=num_heads, hidden_size=hidden, optimization_options=options)
    print('  fusion summary:', optimizer.get_fused_operator_statistics())
    return optimizer.model


def to_fp16(model):
    try:
        from onnxconverter_common import float16
    except ImportError:
        sys.exit('pip install onnxconverter-common')
    # Inputs/outputs stay fp32 so the worker is unchanged; LayerNorm/Softmax stay fp32 for stability.
    return float16.convert_float_to_float16(model, keep_io_types=True, op_block_list=['LayerNormalization', 'Softmax', 'SkipLayerNormalization'])


def check_equivalence(reference, candidate, extra, obs_dim, atol=1e-4, rtol=1e-4):
    """Both graphs must have noise as inputs; compare outputs on random feeds."""
    import onnxruntime as ort
    a = ort.InferenceSession(reference.SerializeToString(), providers=['CPUExecutionProvider'])
    b = ort.InferenceSession(candidate.SerializeToString(), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        feeds = {'obs': rng.standard_normal((1, obs_dim), dtype=np.float32)}
        for name, shape in extra.items():
            feeds[name] = rng.standard_normal(shape, dtype=np.float32)
        ya, yb = a.run(None, feeds)[0], b.run(None, feeds)[0]
        np.testing.assert_allclose(yb, ya, atol=atol, rtol=rtol)
        worst = max(worst, float(np.abs(ya - yb).max()))
    print(f'  max |Δaction| vs. reference over 5 identical random feeds: {worst:.3e}')
    return worst


def write_parts(model, out_dir, name):
    data = model.SerializeToString()
    digest = hashlib.sha256(data).hexdigest()[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i in range(0, len(data), PART_BYTES):
        part = out_dir / f'{name}-{digest}.part{i // PART_BYTES:02d}'
        part.write_bytes(data[i:i + PART_BYTES])
        parts.append(f'assets/{part.name}')
    return parts, len(data)


def count_ops(model):
    from collections import Counter
    c = Counter(n.op_type for n in model.graph.node)
    return len(model.graph.node), c.most_common(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('policy', type=pathlib.Path)
    ap.add_argument('--name', required=True, help='text or goal (used for part file names)')
    ap.add_argument('--obs-dim', type=int, required=True, help='1088 for text, 994 for goal')
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('assets'))
    ap.add_argument('--simplifier', choices=['ort', 'onnxsim'], default='ort', help='provider-independent simplifier (default: ORT basic passes)')
    ap.add_argument('--fuse', action='store_true', help='fuse attention/LayerNorm/GELU into ORT contrib ops')
    ap.add_argument('--num-heads', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--fp16', action='store_true', help='also write an fp16 copy (<name>-fp16) for WebGPU')
    args = ap.parse_args()

    model = onnx.load(str(args.policy))
    print('original:', count_ops(model))

    # Externalize RNG before simplification so equivalence checks use identical noise.
    # This also prevents a constant folder from freezing random samples into weights.
    obs = next(x for x in model.graph.input if x.name == 'obs')
    obs.type.tensor_type.shape.CopyFrom(helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, args.obs_dim]).type.tensor_type.shape)
    print('stage 1: externalize noise')
    model, extra = externalize_noise(model)
    reference = onnx.ModelProto()
    reference.CopyFrom(model)
    print('stage 2: simplify with static shapes')
    model = simplify(model, args.obs_dim, args.simplifier)
    print('after simplify:', count_ops(model))
    check_equivalence(reference, model, extra, args.obs_dim)
    # The transformer optimizer mutates its input graph; retain an independent reference.
    stage2 = onnx.ModelProto()
    stage2.CopyFrom(model)

    if args.fuse:
        print('stage 3: fusion')
        model = fuse(model, args.num_heads, args.hidden)
        print('after fusion:', count_ops(model))
        check_equivalence(stage2, model, extra, args.obs_dim)

    onnx.checker.check_model(model)
    parts, nbytes = write_parts(model, args.out, args.name)
    snippet = {'parts': parts, 'bytes': nbytes, 'sha256': hashlib.sha256(model.SerializeToString()).hexdigest(), 'extra_inputs': extra, 'preoptimized': True}
    if args.fp16:
        print('stage 4: fp16 copy')
        half = to_fp16(model)
        check_equivalence(stage2, half, extra, args.obs_dim, atol=2e-2, rtol=2e-2)
        parts16, nbytes16 = write_parts(half, args.out, f'{args.name}-fp16')
        snippet['fp16'] = {'parts': parts16, 'bytes': nbytes16, 'sha256': hashlib.sha256(half.SerializeToString()).hexdigest()}
    print('\nmanifest.json → policies.%s: merge these keys\n' % args.name)
    print(json.dumps(snippet, indent=2))


if __name__ == '__main__':
    main()
