"""Small deterministic MatMul ONNX for runtime validation only."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['input', 'weight'], ['output'])], 'gpu-smoke',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 64])],
        [numpy_helper.from_array(np.eye(64, dtype=np.float32), 'weight')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    data = np.arange(64, dtype=np.float32).reshape(1, 64) / 64
    np.savez(args.output.with_suffix('.inputs.npz'), input=data)
    np.savez(args.output.with_suffix('.expected.npz'), output=data)
    print(args.output)


if __name__ == '__main__':
    main()
