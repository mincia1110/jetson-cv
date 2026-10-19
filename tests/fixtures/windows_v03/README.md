# Windows v03 reference fixtures

Source: user-supplied `gui_modified_files.7z`, received 2026-10-19.

- `visualad_inferencer_v03.py` and `func_inference_v03.py` retain the supplied source.
- `scoring.py` is the supplied `utils/scoring.py`.
- Only CRLF line endings were normalized to LF.

These files are independent test references, not Jetson runtime imports.
`test_visualad_v03.py` loads them with isolated module names and mocked ONNX
outputs to compare the original wrapper's score, filtered map and threshold
behavior against the Jetson adapter. The original `utils/__init__.py` is not
loaded, so unrelated training/analysis dependencies are not required.
