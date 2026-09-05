"""Execute a saved TensorRT engine on preprocessed inputs, optionally compare outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def compare_outputs(actual, expected, rtol, atol):
    import numpy as np
    if set(actual) != set(expected):
        return {'pass': False, 'error': 'Output names differ.'}
    checks = {}
    for name, value in actual.items():
        ref = expected[name]
        matches = value.shape == ref.shape
        finite = bool(np.all(np.isfinite(value)) and np.all(np.isfinite(ref)))
        checks[name] = {'pass': bool(matches and finite and np.allclose(value, ref, rtol=rtol, atol=atol)),
                        'actual_shape': list(value.shape), 'expected_shape': list(ref.shape)}
    return {'pass': bool(checks) and all(item['pass'] for item in checks.values()), 'outputs': checks}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('engine', type=Path)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--expected', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rtol', type=float, default=1e-4)
    p.add_argument('--atol', type=float, default=1e-5)
    args = p.parse_args()
    if args.rtol < 0 or args.atol < 0:
        p.error('Tolerances must be nonnegative.')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'backend': 'TensorRT', 'pass': False, 'execution_pass': False,
              'comparison': None, 'rtol': args.rtol, 'atol': args.atol}
    try:
        import numpy as np
        import tensorrt as trt
        try:
            from .trt_session import TensorRTSession
        except ImportError:
            from trt_session import TensorRTSession
        report['tensorrt_version'] = trt.__version__
        report['engine_sha256'] = sha256(args.engine)
        report['inputs_sha256'] = sha256(args.inputs)
        with np.load(args.inputs, allow_pickle=False) as saved:
            feed = {name: saved[name] for name in saved.files}
        with TensorRTSession(args.engine) as session:
            start = time.perf_counter()
            actual = session.infer(feed)
            report['wall_time_ms_including_transfers'] = (time.perf_counter() - start) * 1000
        np.savez(args.output / 'outputs.npz', **actual)
        report['execution_pass'] = True
        if args.expected:
            report['expected_sha256'] = sha256(args.expected)
            with np.load(args.expected, allow_pickle=False) as saved:
                expected = {name: saved[name] for name in saved.files}
            report['comparison'] = compare_outputs(actual, expected, args.rtol, args.atol)
        report['pass'] = report['execution_pass'] and (report['comparison'] is None or report['comparison']['pass'])
    except Exception as exc:
        report['error'] = str(exc)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
