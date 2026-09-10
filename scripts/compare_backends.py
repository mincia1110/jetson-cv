"""Compare ONNX Runtime and TensorRT using exactly the same saved inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from inference_backend import ONNXSession
from trt_session import TensorRTSession


def compare(reference, actual, rtol, atol):
    if set(reference) != set(actual):
        raise ValueError('Output names differ')
    result = {}
    for name, expected in reference.items():
        got = actual[name]
        same_shape = expected.shape == got.shape
        finite = bool(np.isfinite(expected).all() and np.isfinite(got).all())
        passed = bool(expected.size and same_shape and finite and np.allclose(expected, got, rtol=rtol, atol=atol))
        result[name] = {'pass': passed, 'reference_shape': list(expected.shape),
                        'actual_shape': list(got.shape), 'reference_dtype': str(expected.dtype),
                        'actual_dtype': str(got.dtype),
                        'max_abs_error': float(np.max(np.abs(expected.astype(float)-got.astype(float))))
                        if same_shape and finite and expected.size else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provider', default='CUDAExecutionProvider')
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--atol', type=float, default=1e-5)
    args = parser.parse_args()
    if not np.isfinite([args.rtol, args.atol]).all() or min(args.rtol, args.atol) < 0:
        parser.error('Tolerances must be finite and nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.inputs, allow_pickle=False) as archive:
        feed = {name: archive[name] for name in archive.files}
    with ONNXSession(args.onnx, args.provider) as session:
        reference = session.infer(feed)
    with TensorRTSession(args.engine) as session:
        actual = session.infer(feed)
    np.savez(args.output/'onnx_outputs.npz', **reference)
    np.savez(args.output/'trt_outputs.npz', **actual)
    report = compare(reference, actual, args.rtol, args.atol)
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report or not all(item['pass'] for item in report.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
