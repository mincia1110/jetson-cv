"""Exercise model selection and switching without a display or Jetson device."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


source = Path(__file__).resolve().parents[1] / 'MOSA_visualAD_comb_jetson.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
nodes = [node for node in tree.body if
         (isinstance(node, ast.FunctionDef) and node.name == 'show_model_selection_gui') or
         (isinstance(node, ast.ClassDef) and node.name == 'CameraGUI')]
namespace = {'messagebox': Mock()}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)


class ModelSwitchTests(unittest.TestCase):
    def setUp(self):
        namespace['messagebox'].reset_mock()
        self.app = namespace['CameraGUI'].__new__(namespace['CameraGUI'])
        self.app.running = True
        self.app._measurement_busy = False
        self.app._model_changing = False
        self.app._model_cleanup_failed = False
        self.app.selected_model_name = 'A'
        self.app.inferencer = Mock()
        for name in ('root', 'result_container', 'result_label', 'model_name_label',
                     'measure_button', 'name_entry', 'threshold_entry', 'folder_button',
                     'model_change_btn'):
            setattr(self.app, name, Mock())

    def test_success_uses_selected_slot_runtime_and_closes_previous_model(self):
        old, new = self.app.inferencer, Mock()
        runtime = {'backend': 'tensorrt', 'engine': 'b.engine'}
        def load(path, config):
            old.close.assert_called_once()
            self.assertIsNone(self.app.inferencer)
            return new
        with patch.dict(namespace, show_model_selection_gui=Mock(return_value=('B', 'b.onnx', None, runtime)),
                        MOSAInferencer=Mock(side_effect=load)):
            self.app.change_model()
            namespace['show_model_selection_gui'].assert_called_once_with(parent=self.app.root)
            namespace['MOSAInferencer'].assert_called_once_with('b.onnx', runtime)
        self.assertIs(self.app.inferencer, new)
        self.assertEqual(self.app.selected_model_name, 'B')
        self.app.model_name_label.config.assert_called_with(text='Model: B')
        old.close.assert_called_once()
        self.assertFalse(self.app._model_changing)
        self.app.measure_button.config.assert_called_with(state='normal')

    def test_cancel_preserves_previous_model(self):
        old = self.app.inferencer
        with patch.dict(namespace, show_model_selection_gui=Mock(return_value=(None, None, None, None)),
                        MOSAInferencer=Mock()):
            self.app.change_model()
            namespace['MOSAInferencer'].assert_not_called()
        self.assertIs(self.app.inferencer, old)
        old.close.assert_not_called()

    def test_failed_load_blocks_measurement_and_allows_retry(self):
        old, new = self.app.inferencer, Mock()
        loader = Mock(side_effect=[RuntimeError('load failed'), new])
        with patch.dict(namespace, show_model_selection_gui=Mock(return_value=('B', 'b.onnx', None, {})),
                        MOSAInferencer=loader):
            self.app.change_model()
            old.close.assert_called_once()
            self.assertIsNone(self.app.inferencer)
            self.assertIsNone(self.app.selected_model_name)
            self.app.measure_button.config.assert_called_with(state='disabled')
            self.app.model_change_btn.config.assert_called_with(state='normal')
            self.app._capture_image_impl = Mock()
            self.app.capture_image()
            self.app._capture_image_impl.assert_not_called()
            self.app.change_model()
        self.assertIs(self.app.inferencer, new)
        self.assertEqual(self.app.selected_model_name, 'B')
        self.app.measure_button.config.assert_called_with(state='normal')

    def test_failed_cleanup_never_loads_new_model_and_requires_restart(self):
        self.app.inferencer.close.side_effect = RuntimeError('cleanup failed')
        with patch.dict(namespace, show_model_selection_gui=Mock(return_value=('B', 'b.onnx', None, {})),
                        MOSAInferencer=Mock()):
            self.app.change_model()
            self.app.change_model()
            namespace['MOSAInferencer'].assert_not_called()
            namespace['show_model_selection_gui'].assert_called_once()
        self.assertTrue(self.app._model_cleanup_failed)
        self.assertIsNone(self.app.inferencer)
        self.app.measure_button.config.assert_called_with(state='disabled')
        self.app.model_change_btn.config.assert_called_with(state='disabled')

    def test_same_name_is_reloaded_because_path_or_runtime_may_change(self):
        with patch.dict(namespace, show_model_selection_gui=Mock(return_value=('A', 'new.onnx', None, {})),
                        MOSAInferencer=Mock(return_value=Mock())):
            self.app.change_model()
            namespace['MOSAInferencer'].assert_called_once_with('new.onnx', {})

    def test_busy_measurement_blocks_switch_and_switch_blocks_measurement(self):
        self.app._measurement_busy = True
        with patch.dict(namespace, show_model_selection_gui=Mock()):
            self.app.change_model()
            namespace['show_model_selection_gui'].assert_not_called()
        self.app._measurement_busy = False
        self.app._model_changing = True
        self.app._capture_image_impl = Mock()
        self.app.capture_image()
        self.app._capture_image_impl.assert_not_called()
        self.app._set_measurement_busy(False)
        self.app.measure_button.config.assert_called_with(state='disabled')

    def test_camera_work_started_during_dialog_defers_switch(self):
        def select(**kwargs):
            self.app._set_measurement_busy(True)
            return 'B', 'b.onnx', None, {}
        with patch.dict(namespace, show_model_selection_gui=select, MOSAInferencer=Mock()):
            self.app.change_model()
            namespace['MOSAInferencer'].assert_not_called()
        self.assertTrue(self.app._measurement_busy)
        self.app.measure_button.config.assert_called_with(state='disabled')

    def test_modal_selection_uses_slot_runtime_and_does_not_quit_main_loop(self):
        dialog, parent = Mock(), Mock()
        dialog.winfo_screenwidth.return_value = 1000
        dialog.winfo_screenheight.return_value = 800
        dialog.winfo_width.return_value = 300
        dialog.winfo_height.return_value = 320
        buttons = []
        def button(*args, **kwargs):
            buttons.append(kwargs['command'])
            return Mock()
        dialog.wait_window.side_effect = lambda: buttons[1]()
        tk = Mock(Toplevel=Mock(return_value=dialog))
        ttk = Mock(Button=button)
        runtime = Mock(return_value={'backend': 'onnx'})
        data = {'model_B_name': 'B', 'model_B_path_vis': 'b.onnx'}
        with patch.dict(namespace, tk=tk, ttk=ttk, read_mosa_data=Mock(return_value=data),
                        model_runtime=runtime):
            result = namespace['show_model_selection_gui'](parent)
        self.assertEqual(result, ('B', 'b.onnx', None, {'backend': 'onnx'}))
        runtime.assert_called_once_with(data, 'B')
        dialog.grab_set.assert_called_once()
        dialog.destroy.assert_called_once()
        dialog.quit.assert_not_called()
        parent.quit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
