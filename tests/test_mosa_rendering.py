"""Compare actual GUI rendering against the frozen pre-optimization code."""
import ast
import datetime
import io
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_renderers():
    namespace = dict(cv2=cv2, np=np, plt=plt, Image=Image, io=io, os=os,
                     time=time, datetime=datetime, img_size_w=448, img_size_h=448,
                     name='sample', print=lambda *args, **kwargs: None)
    reference = ROOT / 'tests/fixtures/mosa_rendering_2203676.py'
    exec(compile(reference.read_text(), str(reference), 'exec'), namespace)

    source = ROOT / 'MOSA_visualAD_comb_jetson.py'
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == '_process_measurement')
    start = next(i for i, node in enumerate(method.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node) == 'pred_score = visual_score')
    end = next(i + 1 for i, node in enumerate(method.body[start:], start)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and node.value.args and isinstance(node.value.args[0], ast.Constant)
               and node.value.args[0].value == '- Time(image 2) : ')
    function = ast.parse('def render_current(black_image, visual_map, visual_score, self):\n    pass').body[0]
    function.body = (ast.parse('start = time.time()').body + method.body[start:end]
                     + ast.parse('return img_show1, img_show2, img_show3, vimg').body)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source), 'exec'), namespace)
    return namespace['render_reference'], namespace['render_current']


REFERENCE, CURRENT = load_renderers()


class RenderingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.addCleanup(plt.close, 'all')
        self.config = SimpleNamespace(save_dir=folder.name, user_threshold=.2, crop_v_offset=0)
        rng = np.random.default_rng(19)
        self.black_image = rng.random((336, 336, 3), dtype=np.float32)
        y, x = np.indices((336, 336), dtype=np.float32)
        self.maps = [np.full((336, 336), -.3, dtype=np.float32),
                     np.sin(x / 31) * np.cos(y / 23),
                     np.where(((x // 30 + y // 24) % 2) == 0, .2, .8).astype(np.float32)]
        style = matplotlib.rc_context({'figure.figsize': (6.4, 4.8), 'figure.dpi': 100,
                                       'savefig.dpi': 'figure', 'savefig.format': 'pdf',
                                       'figure.autolayout': False,
                                       'figure.constrained_layout.use': False})
        style.__enter__()
        self.addCleanup(style.__exit__, None, None, None)

    def assert_same_render(self, amap, score):
        expected = REFERENCE(self.black_image, amap, score, self.config)
        actual = CURRENT(self.black_image, amap, score, self.config)
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
        before_jpg = cv2.imencode('.jpg', cv2.vconcat([expected[2], expected[1]]))
        after_jpg = cv2.imencode('.jpg', cv2.vconcat([actual[2], actual[1]]))
        self.assertTrue(before_jpg[0] and after_jpg[0])
        np.testing.assert_array_equal(before_jpg[1], after_jpg[1])

    def test_gui_pixels_and_saved_jpeg_match_for_ok_ng_boundary_maps_and_crops(self):
        for index, amap in enumerate(self.maps):
            for score in (.199999, .2, .200001):
                for offset in (-.1, 0, .1):
                    with self.subTest(map=index, score=score, offset=offset):
                        self.config.crop_v_offset = offset
                        self.assert_same_render(amap, score)

    def test_rendering_preserves_dpi_background_and_transparent_png_settings(self):
        with matplotlib.rc_context({'figure.dpi': 73, 'savefig.dpi': 91,
                                    'figure.facecolor': '#d5e1f0', 'axes.facecolor': '#ffeedd',
                                    'savefig.transparent': True}):
            self.assert_same_render(self.maps[1], .2)

    def test_heatmap_uses_memory_png_without_reading_or_deleting_image_files(self):
        with patch.object(plt, 'savefig', wraps=plt.savefig) as save, \
                patch.object(cv2, 'imread', side_effect=AssertionError('disk image read')), \
                patch.object(os, 'remove', side_effect=AssertionError('disk image deletion')):
            CURRENT(self.black_image, self.maps[1], .2, self.config)
        target = save.call_args.args[0]
        self.assertIsInstance(target, io.BytesIO)
        self.assertTrue(target.closed)
        self.assertEqual(save.call_args.kwargs,
                         {'format': 'png', 'bbox_inches': 'tight', 'pad_inches': 0,
                          'pil_kwargs': {'compress_level': 0}})

    def test_ok_skips_discarded_contour_work_and_keeps_original_image_alias(self):
        with patch.object(cv2, 'findContours', side_effect=AssertionError('unused OK contours')), \
                patch.object(cv2, 'drawContours', side_effect=AssertionError('unused OK overlay')):
            original, overlay, _, _ = CURRENT(self.black_image, self.maps[1], .1, self.config)
        self.assertIs(overlay, original)

    def test_equal_threshold_remains_ng_and_map_mask_remains_inclusive(self):
        with patch.object(cv2, 'findContours', wraps=cv2.findContours) as contours:
            CURRENT(self.black_image, self.maps[2], .2, self.config)
        contours.assert_called_once()
        np.testing.assert_array_equal(contours.call_args.args[0],
                                      (self.maps[2] >= self.config.user_threshold).astype(np.uint8) * 255)

    def test_failed_memory_decode_aborts_instead_of_displaying_invalid_result(self):
        with patch.object(cv2, 'imdecode', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Heatmap PNG decode failed'):
                CURRENT(self.black_image, self.maps[1], .2, self.config)


if __name__ == '__main__':
    unittest.main()
