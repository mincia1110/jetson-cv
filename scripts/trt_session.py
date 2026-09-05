"""Reusable TensorRT session using NVIDIA Polygraphy's buffer/stream management."""
import os
from pathlib import Path


class TensorRTSession:
    """Use one instance in one worker thread; keep it open across captures."""

    def __init__(self, engine_path):
        self.engine_path = Path(engine_path)
        self.runner = None

    def __enter__(self):
        if self.runner is not None:
            raise RuntimeError('Session is already active.')
        if not self.engine_path.is_file():
            raise FileNotFoundError(self.engine_path)
        # Missing dependencies must fail locally, never trigger a runtime pip install.
        os.environ['POLYGRAPHY_AUTOINSTALL_DEPS'] = '0'
        from polygraphy import config
        config.AUTOINSTALL_DEPS = False
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        runner = TrtRunner(EngineFromBytes(BytesFromPath(str(self.engine_path))))
        runner.activate()
        self.runner = runner
        return self

    def infer(self, feed):
        if self.runner is None:
            raise RuntimeError('Use TensorRTSession as a context manager.')
        import numpy as np
        # Preserve dtype and layout; only make storage contiguous for transfer.
        inputs = {name: np.require(value, requirements=['C']) for name, value in feed.items()}
        outputs = self.runner.infer(inputs, check_inputs=True, copy_outputs_to_host=True)
        # Polygraphy reuses buffers. Give the GUI owned arrays for subsequent frames.
        return {name: np.array(value, copy=True) for name, value in outputs.items()}

    def __exit__(self, exc_type, exc, traceback):
        runner, self.runner = self.runner, None
        if runner is not None:
            runner.deactivate()
