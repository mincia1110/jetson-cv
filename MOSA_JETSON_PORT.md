# 첨부 MOSA 메인 기반 Jetson 포트

파일: MOSA_visualAD_comb_jetson.py. 원본 첨부 파일은 수정하지 않았다. 사용자 요청에 따라 파생본과 연결 모듈을 함께 배포한다. 실제 설정은 local/에 별도 보관한다.

## 실행

기존 MOSA의 data.json, utils/transforms.py 및 기존 의존성(torch/torchvision/matplotlib/Pillow 등)을 그대로 사용한다. 추가 bridge는 이 프로젝트 scripts/에 있다. 파생 main은 이 프로젝트 루트에 둔 채 기존 MOSA 작업 폴더에서 실행한다.

```bash
cd /path/to/original/MOSA
source /path/to/jetson-cv/.venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python /path/to/jetson-cv/MOSA_visualAD_comb_jetson.py
```

`examples/mosa_data_jetson_fields.example.json`의 항목을 기존 `data.json`에 병합한다. 예제는 추가 항목만 있으므로 기존 파일을 덮어쓰지 않는다. 카메라 설정은 기존 MOSA data.json에서 읽는다. 모델 선택은 기존 data.json의 A~E 모델 경로를 그대로 사용한다.

- ONNX: backend=onnx, provider=CUDAExecutionProvider.
- TensorRT: data.json의 backend=tensorrt. A~E 각각 model_A_path_engine부터 model_E_path_engine까지 설정한다. 선택한 항목의 엔진만 사용하며 ONNX 모드에서는 엔진이 없어도 된다.
- 카메라 Brightness/ExposureTime/검사기준/capture_no/reset_flag_en은 data.json만 사용한다. ExposureValue는 선택적인 Windows 기록이며 명령에 사용하지 않는다.

기존 camera_check.json의 검증된 `Brightness`, `ExposureTime`, `BRIGHT_min`, `BRIGHT_max`, `RG_gab`, `capture_no`, `reset_flag_en`, `settle_frames`를 data.json에 옮긴다. 없는 settle_frames는 5가 기본값이다. 기존 모델·저장·GUI 키는 보존한다. runtime의 camera_config 키는 삭제해도 되며 더 이상 읽지 않는다. 메인에서 촬영·조건검사를 확인한 뒤 camera_check.json을 삭제할 수 있다. 독립 카메라 GUI에서 사용 중이라면 해당 설정 파일은 유지한다. 실제 설정은 사내/로컬에만 둔다.
- 제공받은 main은 전체 프레임 평균화이므로 새 포트도 전체 프레임을 평균화한다. 이전 test.py의 ROI만 평균화하는 경로와 다르다.
- 카메라/밝기 검사 실패 시 추론을 실행하지 않는다. CSV의 노출 항목은 요청한 시간이며 실제 readback이라고 기록하지 않는다.
- AE/LED/brightness/고정 노출 외의 DLL AXI/EFLC/렌즈 및 영상 속성 일괄 초기화는 이식하지 않았다. 기존 영상의 색상·감마 등 동일성이 필요하면 해당 Linux 제어값을 별도로 검증해야 한다.
- Subspace 함수와 주석 처리된 추론 블록은 유지. 시작 시 사용하지 않는 Subspace 모델 로드만 주석 처리했다. 따라서 sub 모델 경로가 없어도 VisualAD를 선택할 수 있다.
- 추론은 기존과 같이 GUI 스레드에서 동기 실행한다. TensorRT 세션도 같은 스레드에서 생성·사용·종료한다. 화면 멈춤 개선을 위한 UI 전면 리팩터링은 하지 않았다.
- get_transform은 기존 사내 함수를 그대로 사용한다. 브리지는 이미 정규화된 텐서를 다시 정규화하지 않는다.
- 로컬 구조/연산 테스트만 수행. 실제 사내 자원과 Jetson에서 전체 앱 실행 검증은 필요하다.

## 준비 확인

기존 MOSA 환경의 torch, torchvision, matplotlib 및 이 프로젝트 requirements-inference.txt 의존성이 필요하다.

```bash
python -c "import torch, torchvision, scipy, matplotlib, cv2, onnxruntime; print('imports OK')"
```

실제 모델·data.json·utils 코드는 기존 사내 MOSA 폴더에 그대로 둔다. pull만으로 사내 자원이 생성되지는 않는다.

기존 jetson_runtime.json의 backend/provider/input_name/device/fps를 data.json 최상위에 옮긴다. engines 매핑은 각 model_X_path_engine으로 옮긴다. 이관 후 jetson_runtime.json 및 MOSA_RUNTIME_CONFIG는 제거 가능하다. 상대 경로는 기존과 같이 실행 작업 폴더 기준이다.
