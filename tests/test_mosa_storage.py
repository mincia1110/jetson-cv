"""Storage safety tests; no Jetson, display, camera, sudo or real mount required."""
import ast
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / 'MOSA_visualAD_comb_jetson.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
GUI = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'CameraGUI')
NS = {'os': os, 'messagebox': Mock(), 'filedialog': Mock()}
exec(compile(ast.Module(body=[GUI], type_ignores=[]), str(SOURCE), 'exec'), NS)
CameraGUI = NS['CameraGUI']
DATA = '/mnt/jetson_data'


class StorageGuardTests(unittest.TestCase):
    def setUp(self):
        NS['messagebox'].reset_mock()
        NS['filedialog'].reset_mock()
        self.app = CameraGUI.__new__(CameraGUI)
        self.app.measurement_log = Mock()

    def test_plain_directory_is_not_a_mount(self):
        # Existence alone must never make a missing DATA partition acceptable.
        with patch('os.path.ismount', return_value=False), patch('os.path.isdir', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'JETSON_DATA is not mounted'):
                self.app._require_data_mount(DATA + '/inference_results/sample.png')

    def test_mount_root_itself_is_guarded(self):
        with patch('os.path.ismount', return_value=False):
            with self.assertRaises(RuntimeError):
                self.app._require_data_mount(DATA)

    def test_mounted_data_is_allowed(self):
        with patch('os.path.ismount', return_value=True) as mounted:
            self.app._require_data_mount(Path(DATA) / 'inference_results' / 'sample.png')
            mounted.assert_called_once_with(DATA)

    def test_mount_loss_is_not_cached(self):
        with patch('os.path.ismount', side_effect=[True, False]):
            self.app._require_data_mount(DATA + '/result.png')
            with self.assertRaises(RuntimeError):
                self.app._require_data_mount(DATA + '/result.png')

    def test_explicit_local_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch('os.path.ismount') as mounted:
            self.app._require_data_mount(Path(folder) / 'sample.png')
            self.app._require_data_mount('artifacts/local-results')
            mounted.assert_not_called()

    def test_similar_prefix_is_not_data_partition(self):
        with patch('os.path.ismount') as mounted:
            self.app._require_data_mount(DATA + '_backup/result.png')
            mounted.assert_not_called()

    def test_normalized_path_remains_guarded(self):
        with patch('os.path.ismount', return_value=False):
            with self.assertRaises(RuntimeError):
                self.app._require_data_mount(DATA + '/images/../logs/log.csv')

    def test_symlink_into_data_is_guarded(self):
        with tempfile.TemporaryDirectory() as folder:
            alias = Path(folder) / 'data-link'
            alias.symlink_to(DATA, target_is_directory=True)
            with patch('os.path.ismount', return_value=False):
                with self.assertRaises(RuntimeError):
                    self.app._require_data_mount(alias / 'result.png')

    def test_symlink_escape_from_data_is_rejected(self):
        with patch('os.path.ismount', return_value=True), \
                patch('os.path.realpath', return_value='/home/local-result.png'):
            with self.assertRaisesRegex(RuntimeError, 'resolves outside'):
                self.app._require_data_mount(DATA + '/linked-result.png')

    def test_symlink_followed_by_parent_traversal_still_checks_data_mount(self):
        with tempfile.TemporaryDirectory() as folder:
            alias = Path(folder) / 'data-link'
            alias.symlink_to(DATA + '/images', target_is_directory=True)
            # Keep '..' intact: its parent is on DATA after following the link.
            path = os.path.join(str(alias), '..', 'log.csv')
            with patch('os.path.ismount', return_value=False) as mounted:
                with self.assertRaisesRegex(RuntimeError, 'JETSON_DATA is not mounted'):
                    self.app._require_data_mount(path)
                mounted.assert_called_once_with(DATA)
            with patch('os.path.ismount', return_value=True) as mounted:
                self.app._require_data_mount(path)
                mounted.assert_called_once_with(DATA)

    def config(self, enabled):
        return dict(CSV_path=DATA + '/logs', BRIGHT_min=0, BRIGHT_max=255,
                    RG_gab=18, crop_v_offset=0, threshold=0, sub_shift=0, sub_scale=1,
                    Data_log_flag=enabled, Camera_or_png_flag=0, reset_flag_en=False,
                    AI_result_flag=0, Brightness=16, capture_no=1, ExposureTime='1/8s')

    def test_enabled_csv_checked_at_config_load(self):
        with patch.dict(NS, read_mosa_data=lambda: self.config(1), camera_config=lambda data: data), \
                patch('os.path.ismount', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'JETSON_DATA'):
                self.app.load_json()
        self.app.measurement_log.append.assert_not_called()

    def test_disabled_csv_does_not_require_data(self):
        with patch.dict(NS, read_mosa_data=lambda: self.config(0), camera_config=lambda data: data), \
                patch('os.path.ismount') as mounted:
            self.app.load_json()
            mounted.assert_not_called()
        self.app.measurement_log.close.assert_called_once()

    def test_folder_selection_rejects_unmounted_path_and_keeps_old_selection(self):
        self.app.save_dir = '/tmp/previous'
        self.app.folder_label = Mock()
        NS['filedialog'].askdirectory.return_value = DATA + '/inference_results'
        with patch('os.path.ismount', return_value=False):
            self.app.select_save_folder()
        self.assertEqual(self.app.save_dir, '/tmp/previous')
        self.app.folder_label.config.assert_not_called()
        NS['messagebox'].showerror.assert_called_once()

    def test_folder_selection_accepts_mounted_data(self):
        self.app.save_dir = None
        self.app.folder_label = Mock()
        NS['filedialog'].askdirectory.return_value = DATA + '/inference_results'
        with patch('os.path.ismount', return_value=True):
            self.app.select_save_folder()
        self.assertEqual(self.app.save_dir, DATA + '/inference_results')
        NS['messagebox'].showerror.assert_not_called()

    def test_capture_blocks_before_camera_or_inference_and_releases_busy_state(self):
        app = self.app
        app._measurement_busy = False
        app._model_changing = False
        app.running = True
        app.inferencer = Mock()
        app._camera_future = None
        app._set_measurement_busy = Mock()
        app._show_camera_failure = Mock()
        app.show_wait_status = Mock()
        app.root = Mock()
        app.save_dir = DATA + '/inference_results'
        app.name_entry = Mock()
        app.name_entry.get.return_value = 'sample'
        app.threshold_entry = Mock()
        app.threshold_entry.get.return_value = '5'
        app.frame_lock = threading.Lock()
        app.load_json = Mock()
        app._start_camera_measurement = Mock()
        app._process_measurement = Mock()
        with patch('os.path.ismount', return_value=False):
            app.capture_image()
        app._start_camera_measurement.assert_not_called()
        app._process_measurement.assert_not_called()
        app._set_measurement_busy.assert_called_with(False)
        NS['messagebox'].showerror.assert_called_once()

    def test_mount_lost_during_camera_wait_blocks_raw_image_write(self):
        self.app.Data_log_flag = 0
        self.app.Camera_or_png_flag = 1
        cv2 = Mock()
        with patch.dict(NS, cv2=cv2), patch('os.path.ismount', return_value=False):
            with self.assertRaises(RuntimeError):
                self.app._process_measurement('sample', DATA + '/sample.png', 0, object())
        cv2.imwrite.assert_not_called()

    def test_missing_csv_mount_blocks_even_when_image_path_is_local(self):
        self.app.Data_log_flag = 1
        self.app.CSV_path = DATA + '/logs'
        self.app.Camera_or_png_flag = 1
        cv2 = Mock()
        with patch.dict(NS, cv2=cv2), patch('os.path.ismount', return_value=False):
            with self.assertRaises(RuntimeError):
                self.app._process_measurement('sample', '/tmp/sample.png', 0, object())
        cv2.imwrite.assert_not_called()
        self.app.measurement_log.append.assert_not_called()

    def combined_save_block(self):
        method = next(n for n in GUI.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_process_measurement')
        branch = next(n for n in method.body if isinstance(n, ast.If)
                      and any(isinstance(child, ast.Assign)
                              and isinstance(child.targets[0], ast.Name)
                              and child.targets[0].id == 'filepath_combined'
                              for child in n.orelse))
        return compile(ast.Module(body=branch.orelse, type_ignores=[]), str(SOURCE), 'exec')

    def test_result_image_rechecks_mount_after_inference(self):
        self.app.save_dir = DATA + '/inference_results'
        cv2 = Mock()
        env = dict(os=os, self=self.app, cv2=cv2, name='sample',
                   img_show2=object(), img_show3=object())
        with patch('os.path.ismount', return_value=False):
            with self.assertRaises(RuntimeError):
                exec(self.combined_save_block(), env)
        cv2.imwrite.assert_not_called()

    def test_result_image_write_failure_is_reported(self):
        self.app.save_dir = DATA + '/inference_results'
        cv2 = Mock()
        cv2.imwrite.return_value = False
        env = dict(os=os, self=self.app, cv2=cv2, name='sample',
                   img_show2=object(), img_show3=object())
        with patch('os.path.ismount', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'Image write failed'):
                exec(self.combined_save_block(), env)

    def test_csv_rechecks_mount_immediately_before_append(self):
        method = next(n for n in GUI.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_process_measurement')
        block = next(n for n in method.body if isinstance(n, ast.If)
                     and any(isinstance(child, ast.Assign)
                             and isinstance(child.targets[0], ast.Name)
                             and child.targets[0].id == 'csvfilepath'
                             for child in n.body))
        self.app.CSV_path = DATA + '/logs'
        self.app.ExposureTime = '1/8s'
        self.app.camera_report = {}
        with patch('os.path.ismount', return_value=False):
            with self.assertRaises(RuntimeError):
                exec(compile(ast.Module(body=block.body, type_ignores=[]), str(SOURCE), 'exec'),
                     dict(os=os, self=self.app))
        self.app.measurement_log.append.assert_not_called()


if __name__ == '__main__':
    unittest.main()
