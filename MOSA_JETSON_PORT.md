# 첨부 MOSA 메인 기반 Jetson 포트

파일: MOSA_visualAD_comb_jetson.py. 원본 첨부 파일은 수정하지 않았다. 사용자 요청에 따라 파생본과 연결 모듈을 함께 배포한다. 실제 설정은 local/에 별도 보관한다.

## 2026-10-14 Windows 261013 업데이트 반영

비교 기준은 첨부 `MOSA_visualAD_comb_261013.py`와 기존 Jetson 메인이다. 이전 Windows 원본 전체가 없으므로 Windows 버전 사이의 완전한 변경 이력은 아니다.

- 카메라 원본은 `{name}.png`로 저장한다. 덮어쓰기 확인은 촬영 모드에만 표시한다. 파일 모드는 같은 이름의 PNG를 우선 읽고 없으면 BMP를 읽으며, 둘 다 없으면 추론 전에 경고한다. 표시와 CSV에는 실제 읽은 파일명을 기록한다. 결과 합성 이미지는 기존 `{name}_resulf.jpg`를 사용한다.
- 새 설정 `Camera_or_png_flag`를 읽는다(1: 촬영, 0: 파일 입력). 기존 `Camera_or_BMP_flag`도 지원하며 두 키가 있으면 새 키가 우선한다.
- 일반 VisualAD 출력의 TopK ratio를 사용자 전달값 `0.00005`로 변경했다. 336×336 맵의 상위 6픽셀 평균이며 이전 값 `0.005`의 565픽셀보다 100배 작은 비율이다. 첨부 GUI는 `func_inference`를 불러오며 ratio 자체는 정의하지 않으므로 새 Windows 후처리 구현은 첨부 파일만으로 직접 확인할 수 없다.
- 새 GUI에서 제거한 전처리 7×7 Gaussian blur를 메인, 공통 비교 도구와 카메라 밝기/RG 검사에서도 제거했다. 일반 VisualAD 후처리의 Gaussian sigma=4는 계속 표시 맵·마스크에 적용한다.
- 모델이 `fused_anomaly_map`을 출력하면 이미 합성된 맵을 그대로 사용하고 최댓값을 점수로 쓴다. 이 경로는 TopK와 추가 Gaussian 처리를 사용하지 않는다. 새 Windows 코드와 같이 `pred_label`과 `pred_mask`는 0 초과 기준이며 GUI의 OK/NG·contour는 사용자 threshold 기준이다. `fusion_weights`는 읽을 수 있지만 Windows처럼 판정에 사용하지 않는다.
- Windows의 매 측정 CPU `InferenceSession` 생성은 Jetson의 기존 ONNX CUDA/TensorRT 세션을 재사용하는 형태로 반영했다. 두 출력 형식은 공통 후처리에서 선택하므로 측정당 추론은 한 번이다. 실제 fused 모델을 TensorRT로 쓰려면 그 모델에 맞는 엔진이 필요하다.
- Running 표시 전에 큰 글꼴로 상태 영역을 덮는 변경을 반영했다. Windows의 별도 결과창은 빈 `fused_maps`를 heatmap에 넘기는 코드로 변경돼 있지만, Jetson 결과창은 실제 `visual_map`을 표시한다.

Jetson에는 **메인과 `scripts/`를 함께** 갱신하고 GUI를 재시작한다. 기존 `data.json`과 모델은 별도로 유지한다. 비율과 전처리가 모두 바뀌었으므로 같은 PNG/BMP로 Windows·Jetson 점수와 threshold를 다시 비교한다. 이 작업에서는 로컬 회귀 검증을 수행하며 실제 장치 촬영·CUDA/TensorRT 실행 검증은 별도로 필요하다.

검증(2026-10-14, Mac의 `/tmp/jetson-cv-261013-check` 전용 venv):

- `/tmp/jetson-cv-261013-check/bin/python -m unittest discover -s tests -v`: 71개 통과, skip 없음. PNG 원본의 저장·재읽기 픽셀 보존, PNG/BMP 선택과 덮어쓰기 취소, 새/기존 설정 키, blur 없는 전처리·밝기 검사, 일반 TopK 및 fused 맵 후처리, 실제 CPU ONNX 합성 모델과 세션 재사용을 확인했다. TensorRT 경로는 mock으로 연결을 확인했다.
- `/tmp/jetson-cv-261013-check/bin/python -m compileall -q MOSA_visualAD_comb_jetson.py scripts tests examples`: 종료 코드 0. 기존 미사용 Windows DLL 경로 문자열에서 Python 3.12의 invalid escape SyntaxWarning이 발생했다.
- `/tmp/jetson-cv-261013-check/bin/python -m pip check`: `No broken requirements found.`
- `git diff --check`: 통과.

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
- 카메라 Brightness/ExposureTime/검사기준/capture_no/reset_flag_en은 data.json만 사용한다. fixed 모드에서 ExposureValue는 Windows 기록이다. windows_800/windows_800_full은 관측된 800 명령 시퀀스를 사용한다. [호환 모드 안내](WINDOWS_800_CHECK.md)를 참조한다.

기존 camera_check.json의 검증된 `Brightness`, `ExposureTime`, `BRIGHT_min`, `BRIGHT_max`, `RG_gab`, `capture_no`, `reset_flag_en`, `settle_frames`를 data.json에 옮긴다. 없는 settle_frames는 5가 기본값이다. 기존 모델·저장·GUI 키는 보존한다. runtime의 camera_config 키는 삭제해도 되며 더 이상 읽지 않는다. 메인에서 촬영·조건검사를 확인한 뒤 camera_check.json을 삭제할 수 있다. 독립 카메라 GUI에서 사용 중이라면 해당 설정 파일은 유지한다. 실제 설정은 사내/로컬에만 둔다.
- 제공받은 main은 전체 프레임 평균화이므로 새 포트도 전체 프레임을 평균화한다. 이전 test.py의 ROI만 평균화하는 경로와 다르다.
- 카메라/밝기 검사 실패 시 추론을 실행하지 않는다. CSV의 노출 항목은 요청한 시간이며 실제 readback이라고 기록하지 않는다.
- 영상 속성은 아래 video_controls로 설정한다. DLL AXI/EFLC/렌즈 API의 완전한 대체는 아니다. windows_800_full은 캡처된 미확인 쓰기도 재생하는 실험용 모드이며 실제 영상 동등성을 별도로 검증해야 한다.
- Subspace 함수와 주석 처리된 추론 블록은 유지. 시작 시 사용하지 않는 Subspace 모델 로드만 주석 처리했다. 따라서 sub 모델 경로가 없어도 VisualAD를 선택할 수 있다.
- 추론은 기존과 같이 GUI 스레드에서 동기 실행한다. TensorRT 세션도 같은 스레드에서 생성·사용·종료한다. 화면 멈춤 개선을 위한 UI 전면 리팩터링은 하지 않았다.
- get_transform은 기존 사내 함수를 그대로 사용한다. 브리지는 이미 정규화된 텐서를 다시 정규화하지 않는다.
- 일반 VisualAD 점수는 Gaussian 적용 전 합산 map의 상위 0.005% 평균이다. Windows 261013 업데이트에 따라 비율 0.00005를 사용한다(336×336에서는 ceil로 6픽셀). fused 출력 모델은 맵의 최댓값을 사용한다. 이전 0.005에서 확인한 동일 BMP 일치는 과거 검증이며 새 버전은 다시 비교해야 한다.
- 로컬 구조/연산 테스트만 수행. 실제 사내 자원과 Jetson에서 전체 앱 실행 검증은 필요하다.

## 준비 확인

기존 MOSA 환경의 torch, torchvision, matplotlib 및 이 프로젝트 requirements-inference.txt 의존성이 필요하다.

```bash
python -c "import torch, torchvision, scipy, matplotlib, cv2, onnxruntime; print('imports OK')"
```

실제 모델·data.json·utils 코드는 기존 사내 MOSA 폴더에 그대로 둔다. pull만으로 사내 자원이 생성되지는 않는다.

기존 jetson_runtime.json의 backend/provider/input_name/device/fps를 data.json 최상위에 옮긴다. engines 매핑은 각 model_X_path_engine으로 옮긴다. 이관 후 jetson_runtime.json 및 MOSA_RUNTIME_CONFIG는 제거 가능하다. 상대 경로는 기존과 같이 실행 작업 폴더 기준이다.

자동복구는 data.json의 `reset_flag_en: true` 또는 1로 활성화한다. 조건 실패 시 FAIL(WAIT) → 스트림 닫기 → 1초 대기 → 재초기화 → READY(회색) → 수동 재측정 순서다. 리셋 완료 뒤에는 after 진단 통과 여부와 무관하게 해당 클릭의 추론/결과 저장을 중단한다. READY는 영상 검사 통과가 아니라 재초기화 완료이며, 다음 측정에서 새 프레임을 검사한다. 재초기화 자체가 실패하면 FAIL로 남는다. 카메라 대기는 worker와 Tk polling으로 처리하여 WAIT를 표시한다. 전체 before/after, reset_succeeded, manual_retry_required, allow_inference 보고서는 터미널에 출력한다.

MOSA의 windows_800/windows_800_full은 설정이 같으면 측정 전 제어값을 다시 쓰지 않는다(settings_action=preserved). 바뀐 카메라 설정은 다음 측정에서 적용한다. fixed 모드와 독립 카메라 진단 GUI는 매회 재적용 정책을 유지한다. 노출 readback은 여전히 unavailable이다.

측정 log.csv의 마지막 열 backend에 실제 로드한 onnx/tensorrt를 기록한다. 기존 형식의 로그는 처음 기록할 때 backend 열을 추가하며 이전 행은 unknown으로 보존한다. 실행 중 data.json의 backend만 바꿔도 이미 로드한 추론기는 바뀌지 않으므로 전환 시 앱을 재시작한다.

## 영상 속성 재현

`examples/mosa_data_jetson_fields.example.json`의 `video_controls` 객체를 기존 data.json에 병합한다. 밝기는 기존 최상위 Brightness를 사용한다. fixed 모드에서 객체를 생략하면 추가 영상 속성을 설정하지 않는다. Windows 프로필은 기본값을 채운다. 최초 시작·설정 변경·재연결 복구 시 적용하며, fixed 모드에서는 매 측정에도 적용한다. 자동 화이트밸런스를 먼저 끈 뒤 색온도를 설정하고 모든 지정 항목을 readback 검증한다. 변경은 다음 측정부터 반영된다. fixed 모드에서 항목을 삭제하면 기존 장치 값을 되돌리지는 않는다.

AM7115MZT 사용자 제공 범위로 검증한다. 예제의 대비16/채도32/색조0/감마5/선명도0/색온도5800은 기존 Windows 코드의 요청값이다. 실제 Windows readback 및 이미지 동등성은 별도 검증한다. power_line_frequency=2는 현재 Jetson의 60Hz 설정을 유지하는 값이며 LED 제어가 아니다. ColorEnable/BacklightCompensation/Gain은 해당 장치 UVC 목록에 없어 보내지 않는다. focus_absolute도 inactive 상태이므로 이번 영상 속성 설정에 포함하지 않는다.
