"""Apply the SAME VisualAD postprocessing to saved ONNX/TRT outputs; no images needed."""
import argparse
import json
from pathlib import Path
import numpy as np
from visualad_adapter import postprocess
from compare_backends import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--threshold', type=float, required=True, help='Internal threshold (GUI value / 10)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rtol', type=float, default=1e-4)
    p.add_argument('--atol', type=float, default=1e-5)
    args = p.parse_args()
    if not np.isfinite([args.threshold, args.rtol, args.atol]).all() or min(args.rtol,args.atol)<0:
        p.error('Finite threshold and nonnegative finite tolerances required')
    results = []
    for path in (args.reference, args.actual):
        with np.load(path, allow_pickle=False) as archive:
            results.append(postprocess({k: archive[k] for k in archive.files}, args.threshold, {}))
    ref, got = results
    tensors = compare({'filtered_map':ref['anomaly_map'], 'score':np.array(ref['pred_score'])},
                      {'filtered_map':got['anomaly_map'], 'score':np.array(got['pred_score'])},args.rtol,args.atol)
    report = {'threshold':args.threshold, 'numerical':tensors,
              'reference_score':ref['pred_score'], 'actual_score':got['pred_score'],
              'score_abs_error':abs(ref['pred_score']-got['pred_score']),
              'reference_decision':'OK' if ref['pred_score']<args.threshold else 'NG',
              'actual_decision':'OK' if got['pred_score']<args.threshold else 'NG',
              'mask_mismatch_pixels':int(np.count_nonzero(ref['pred_mask'] != got['pred_mask']))}
    report['decision_match'] = report['reference_decision']==report['actual_decision']
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['decision_match'] or not all(v['pass'] for v in tensors.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
