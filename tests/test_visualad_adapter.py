import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    import cv2
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Needs torch scipy OpenCV')
class VisualADTests(unittest.TestCase):
    def test_matches_supplied_wrapper(self):
        from visualad_adapter import postprocess
        rng = np.random.default_rng(18)
        outputs = {name:rng.normal(size=(1,1024)).astype(np.float32) for name in
                   ('anomaly_features_enhanced','normal_features_enhanced')}
        for layer in (6,12,18,24):
            outputs[f'patch_tokens_transformed_layer_{layer}'] = rng.normal(size=(1,579,1024)).astype(np.float32)
        # Reference transliteration of the supplied wrapper, including per-layer
        # feature normalization and cosine_similarity's own normalization.
        maps=[]
        for layer in (6,12,18,24):
            a=F.normalize(torch.from_numpy(outputs['anomaly_features_enhanced']),dim=1,eps=1e-8)
            n=F.normalize(torch.from_numpy(outputs['normal_features_enhanced']),dim=1,eps=1e-8)
            t=F.normalize(torch.from_numpy(outputs[f'patch_tokens_transformed_layer_{layer}'])[:,3:,:],dim=2,eps=1e-8)
            v=torch.cosine_similarity(t,a.unsqueeze(1),dim=2)-torch.cosine_similarity(t,n.unsqueeze(1),dim=2)
            maps.append(F.interpolate(torch.nan_to_num(v,nan=0.).reshape(1,1,24,24),size=(336,336),mode='bilinear',align_corners=False).squeeze(1))
        raw=torch.stack(maps).sum(0)[0]
        score=torch.topk(raw.reshape(-1),6).values.mean().item()  # ceil(336*336*0.00005)
        result=postprocess(outputs,score,{})
        np.testing.assert_array_equal(result['anomaly_map'],gaussian_filter(raw.numpy(),sigma=4))
        self.assertEqual(result['pred_score'],score)
        self.assertFalse(result['pred_label']) # source uses strict >
        changed={k:v.copy() for k,v in outputs.items()}
        for k in changed:
            if k.startswith('patch_'):
                changed[k][:,:3,:]=10000
        np.testing.assert_array_equal(result['anomaly_map'],postprocess(changed,score,{})['anomaly_map'])
        del changed['patch_tokens_transformed_layer_24']
        with self.assertRaises(ValueError): postprocess(changed,score,{})

    def test_fused_export_matches_windows_max_score_and_zero_mask(self):
        from visualad_adapter import postprocess
        amap = np.full((336, 336), -0.1, dtype=np.float32)
        amap[0, 0], amap[0, 1] = 0.8, 0
        for shape in ((336, 336), (1, 336, 336), (1, 1, 336, 336)):
            outputs = {'fused_anomaly_map': amap.reshape(shape),
                       'fusion_weights': np.array([0.5, 0.5], dtype=np.float32)}
            result = postprocess(outputs, 1.0, {})
            np.testing.assert_array_equal(result['anomaly_map'], amap)
            self.assertEqual(result['pred_score'], float(amap.max()))
            self.assertEqual(result['pred_label'], 1)  # Windows uses zero, not GUI threshold.
            np.testing.assert_array_equal(result['pred_mask'], (amap > 0).astype(np.float32))
            result['anomaly_map'][0, 0] = 5
            self.assertEqual(amap[0, 0], np.float32(0.8))
        result = postprocess({'fused_anomaly_map': np.zeros_like(amap)}, -1, {})
        self.assertEqual(result['pred_label'], 0)
        self.assertFalse(result['pred_mask'].any())

    def test_invalid_fused_map_is_rejected(self):
        from visualad_adapter import postprocess
        for amap in (np.zeros((2, 336, 336), np.float32),
                     np.zeros((336, 336), np.float64),
                     np.full((336, 336), np.nan, np.float32)):
            with self.assertRaisesRegex(ValueError, 'fused_anomaly_map'):
                postprocess({'fused_anomaly_map': amap}, 0, {})

    def test_preprocess_normalizes_once(self):
        from visualad_adapter import preprocess, MEAN, STD
        from measurement_pipeline import visible_preprocess
        frame=np.zeros((1944,2592,3),np.uint8)
        frame[900:1100]=255
        feed,display=preprocess(frame,{})
        expected=np.asarray(visible_preprocess(frame),dtype=np.float32)/255
        expected=(expected-np.array(MEAN,np.float32))/np.array(STD,np.float32)
        np.testing.assert_array_equal(feed['input'],expected.transpose(2,0,1)[None])
        self.assertEqual(feed['input'].dtype,np.float32)
        self.assertEqual(display.shape,(336,336,3))

    def test_visible_preprocess_matches_unblurred_windows_image(self):
        from measurement_pipeline import visible_preprocess
        frame = np.zeros((1944, 2592, 3), np.uint8)
        frame[900:1100:2, ::2] = [20, 120, 255]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        normalized = cv2.normalize(rgb, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        crop = np.power(np.clip((normalized[900:1100, :2590] - 0.3) / 0.7, 0, 1), 1.4)
        expected = np.zeros((336, 336, 3), np.float32)
        expected[84:252] = cv2.resize(crop, (336, 168))
        np.testing.assert_array_equal(np.asarray(visible_preprocess(frame)), (expected * 255).astype(np.uint8))


if __name__=='__main__': unittest.main()
