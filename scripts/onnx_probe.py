"""Inspect ONNX I/O or execute saved input tensors with explicit provider."""
import argparse
import json
from pathlib import Path
import numpy as np
from inference_backend import ONNXSession


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--provider', default='CUDAExecutionProvider')
    p.add_argument('--inputs', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with ONNXSession(args.model, args.provider) as backend:
        session = backend.session
        report = {'providers': session.get_providers(),
                  'inputs': [{'name': x.name, 'shape': x.shape, 'type': x.type} for x in session.get_inputs()],
                  'outputs': [{'name': x.name, 'shape': x.shape, 'type': x.type} for x in session.get_outputs()]}
        if args.inputs:
            with np.load(args.inputs, allow_pickle=False) as archive:
                outputs = backend.infer({key: archive[key] for key in archive.files})
            if any(not np.isfinite(v).all() for v in outputs.values()):
                raise ValueError('Non-finite outputs')
            np.savez(args.output/'outputs.npz', **outputs)
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
