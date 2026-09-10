"""Shared GUI/CLI path. Load existing model preprocessing/postprocessing locally."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from inference_backend import open_backend


def load_adapter(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('measurement_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('preprocess', 'postprocess'):
        if not callable(getattr(module, name, None)):
            raise ValueError(f'Adapter must define {name}')
    return module


def visible_preprocess(frame):
    """Only the explicit test.py operations before its unavailable get_transform."""
    import cv2
    from PIL import Image
    if frame.shape != (1944, 2592, 3) or frame.dtype != np.uint8:
        raise ValueError('Expected uint8 2592x1944 BGR image')
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.GaussianBlur(rgb, (7, 7), 0)
    normalized = cv2.normalize(rgb, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    crop = normalized[900:1100, 0:2590]
    crop = np.power(np.clip((crop - 0.3) / 0.7, 0, 1), 1.4)
    crop = cv2.resize(crop, (336, 168))
    padded = np.zeros((336, 336, 3), dtype=np.float32)
    padded[84:252] = crop
    return Image.fromarray((padded * 255).astype(np.uint8))


class Pipeline:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        self.adapter = load_adapter(self.config['adapter'])
        self.backend = open_backend(self.config)
        self.backend.__enter__()
        return self

    def run(self, frame, threshold):
        if not np.isfinite(threshold):
            raise ValueError('Threshold must be finite')
        start = time.perf_counter()
        feed, display_rgb = self.adapter.preprocess(frame, self.config)
        feed = {k: np.array(v, copy=True) for k, v in feed.items()}
        if not feed or any(not np.isfinite(v).all() for v in feed.values()):
            raise ValueError('Invalid input tensors')
        prepared = time.perf_counter()
        outputs = self.backend.infer(feed)
        inferred = time.perf_counter()
        if any(not np.isfinite(v).all() for v in outputs.values()):
            raise ValueError('Non-finite model output')
        result = self.adapter.postprocess(outputs, threshold, self.config)
        amap = np.asarray(result['anomaly_map'])
        score = float(result['pred_score'])
        if amap.ndim != 2 or not np.isfinite(amap).all() or not np.isfinite(score):
            raise ValueError('Adapter must return finite 2D anomaly_map and scalar pred_score')
        display_rgb = np.asarray(display_rgb)
        if display_rgb.dtype != np.uint8 or display_rgb.shape != (*amap.shape, 3):
            raise ValueError('Display must be uint8 RGB aligned with anomaly_map')
        result.update(anomaly_map=amap, pred_score=score, display_rgb=display_rgb,
                      decision='OK' if score < threshold else 'NG', feed=feed, outputs=outputs,
                      timings={'preprocess': prepared-start, 'inference': inferred-prepared,
                               'total': time.perf_counter()-start})
        return result

    def __exit__(self, *args):
        return self.backend.__exit__(*args)


def save_measurement(folder, frame, result, config, threshold, camera_report):
    import cv2
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(folder/'original.bmp'), frame):
        raise RuntimeError('Image save failed')
    np.savez(folder/'inputs.npz', **result['feed'])
    np.savez(folder/'outputs.npz', **result['outputs'])
    np.save(folder/'anomaly_map.npy', result['anomaly_map'])
    report = {'config': config, 'threshold': threshold, 'camera': camera_report,
              'score': result['pred_score'], 'decision': result['decision'], 'timings': result['timings']}
    (folder/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
