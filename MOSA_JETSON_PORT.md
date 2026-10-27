# 첨부 MOSA 메인 기반 Jetson 포트

파일: MOSA_visualAD_comb_jetson.py. 원본 첨부 파일은 수정하지 않았다. 사용자 요청에 따라 파생본과 연결 모듈을 함께 배포한다. 실제 설정은 local/에 별도 보관한다.

## 2026-10-19 Windows v03 업데이트 반영

첨부 `gui_modified_files.7z`의 `MOSA_visualAD_comb.py`, `func_inference_v03.py`, `visualad_inferencer_v03.py`, `utils/scoring.py`를 기준으로 추론 경로를 갱신했다. 아래 2026-10-14 절은 이전 버전의 이력이며, fused 맵 최댓값·0 기준 판정과 네 레이어 합산은 현재 경로에 적용되지 않는다.

- fused 맵과 토큰 맵 모두 **336×336 raw 맵 → TopK ratio 0.00005의 평균 점수 → sigma=4 Gaussian 표시 맵** 순서다. 점수는 6픽셀 평균이며 Gaussian 적용 전에 계산한다. `pred_label`은 점수 > 사용자 threshold, `pred_mask`는 필터된 맵 > 사용자 threshold다. GUI의 OK/NG는 기존처럼 score < threshold일 때 OK이며 동일값은 NG다.
- fused 출력은 `[H,W]`, `[1,H,W]`, `[1,1,H,W]`를 지원한다. float16/32/64 출력을 float32로 바꾸고 크기가 다르면 bilinear, `align_corners=False`로 입력 크기에 맞춘 뒤 점수를 계산한다.
- 토큰 경로는 `[6,12,18,24]` 중 마지막으로 존재하는 레이어 하나만 사용한다. 앞의 특수 토큰 3개를 제외하고 정사각형 패치 맵을 만든다. enhanced 특징 쌍이 없으면 `anomaly_features`/`normal_features` 쌍을 사용한다.
- MOSA의 ONNX 실행은 fused 출력이 있으면 `fused_anomaly_map`만 요청한다. TensorRT는 Polygraphy 0.49.9의 GPU 출력에서 필요한 fused 맵만 CPU로 복사한다. TensorRT 엔진의 다른 출력 바인딩과 GPU 계산·할당 자체가 제거되는 것은 아니다. fused 출력이 없으면 토큰 출력을 사용하며, 별도 raw 비교/진단 도구는 기본적으로 전체 출력을 유지한다.
- `utils/scoring.py`의 점수 계산은 `scripts/visualad_scoring.py`로 옮겼으며 `visualad_adapter.py`에서 불러온다. **현재 ratio 정의 위치는 `scripts/visualad_scoring.py`다.** 외부 사내 `utils` 패키지를 덮어쓰지 않아도 새 후처리가 적용된다.
- 메인의 미사용 Subspace 생성자 인자와 측정용 빈 리스트를 정리했다. Windows v03의 CPU 고정 로더 대신 Jetson의 CUDA/TensorRT 세션 재사용을 유지한다. Windows 결과창의 빈 `fused_maps` 참조는 가져오지 않고 실제 `visual_map`을 표시한다.

`data.json`에 새 필수 키는 없다. 새 Combined ONNX로 바꾸는 경우 선택한 `model_X_path_vis`를 지정하고, TensorRT는 해당 ONNX로 만든 엔진의 `model_X_path_engine`도 지정한 뒤 재시작한다. `threshold`는 같은 기준 영상으로 Windows v03과 다시 비교한다. PNG 저장, PNG 우선·BMP 대체 읽기 및 기존 카메라 설정은 계속 지원한다.

교체 파일은 `MOSA_visualAD_comb_jetson.py`와 `scripts/` 전체이며 새 `visualad_scoring.py`가 반드시 포함돼야 한다. 실제 장치의 카메라·CUDA/TensorRT 실행과 RAM 사용량은 별도 확인 대상이다. 원본 v03 추론기·wrapper·scoring을 `tests/fixtures/windows_v03/`에 보관해 수치 비교에 사용한다. 학습·분석용 나머지 utils 파일과 Windows Python 3.11의 `__pycache__`는 Jetson 실행 파일로 복사하지 않는다.

검증(2026-10-19, Mac의 `/tmp/jetson-cv-v03-check` 전용 venv):

- `/tmp/jetson-cv-v03-check/bin/python -m unittest discover -s tests -v`: **80개 통과, skip 없음**. 원본 v03 wrapper와 합성 출력의 맵·점수·마스크·label 일치, 작은 fused 맵의 보간, float16/32/64 처리, enhanced/plain 특징, 부분 레이어·마지막 레이어 선택, TopK·Gaussian 순서와 threshold 경계를 확인했다. 실제 CPU ONNX 실행과 ONNX 출력 선택, TensorRT DeviceView의 선택 복사(mock)도 확인했다.
- `/tmp/jetson-cv-v03-check/bin/python -m compileall -q MOSA_visualAD_comb_jetson.py scripts tests examples`: 종료 코드 0. 기존 Windows DLL 경로의 invalid escape SyntaxWarning은 유지된다.
- `/tmp/jetson-cv-v03-check/bin/python -m pip check`: `No broken requirements found.`
- `git diff --check`: 통과.

## 파일 입력 모드와 실행 중 전환

`data.json`의 `Camera_or_png_flag`가 0이면 PNG/BMP 파일 입력, 1이면 촬영이다. 기존 `Camera_or_BMP_flag`도 지원하며 두 키가 함께 있으면 새 키가 우선한다.

- 파일 모드로 시작하면 카메라 객체를 생성하지 않고 장치 연결·초기화·미리보기를 생략한다. 메인 영상 영역에 파일 모드를 표시하며, 카메라 없이도 파일 추론과 OK/NG 표시, 정상 종료가 가능하다. 기존 영상 밝기/RG 검사와 설정 검증은 유지한다.
- 실행 중 설정을 바꾸면 다음 측정에서 다시 읽는다. 파일 모드에서 촬영 모드로 바꾼 첫 측정은 worker의 카메라 연결·초기화를 Tk polling으로 기다린 뒤 같은 측정을 이어간다. GUI 재시작이나 모델 재로드는 필요 없다.
- 연결 실패 시 해당 측정만 중단하고 GUI를 유지한다. 파일 모드로 돌아가 추론하거나, 장치를 연결한 뒤 촬영을 재시도할 수 있다.
- 한 번 연결한 카메라와 미리보기는 앱 종료까지 유지·재사용한다. 파일 모드로 돌아가면 촬영과 카메라 조건 검사는 요청하지 않으며 기존 PNG 우선·BMP 대체 입력 경로를 사용한다. 전환 중 진행 중인 측정이 끝나기 전에는 다음 측정을 시작하지 않는다.

추가 설정 키는 없다. CUDA/TensorRT 세션, 전처리와 v03 후처리·점수·threshold 규칙은 그대로 재사용한다.

검증(2026-10-19): `/tmp/jetson-cv-v03-check/bin/python -B -m unittest discover -s tests -v`에서 **90개 통과, skip 없음**. 파일 모드의 무카메라 시작, 비동기 연결 대기·실패·재시도, 파일↔촬영 전환과 모델 재사용, OK/NG 표시, 연결 대기 중 종료와 최종 정리를 mock으로 확인했다. 실제 Jetson의 카메라·GUI·CUDA 동작은 별도 확인이 필요하다.

## 실행 중 모델 변경

메인 화면의 **모델 변경** 버튼에서 `data.json`의 A~E 항목을 다시 선택한다. 선택 시 파일을 다시 읽으며 ONNX 경로와 backend/provider/input_name, TensorRT 모드라면 해당 슬롯의 엔진 경로를 적용한다. 이름이 같아도 경로나 설정을 바꿨을 수 있으므로 다시 로드한다. 취소하거나 새 모델 로드에 실패하면 기존 모델을 유지한다. 측정 또는 카메라 작업 중에는 변경할 수 없다. 로드는 기존 추론기와 동일한 GUI 스레드에서 수행하므로 로딩 중 화면이 잠시 멈출 수 있으며, 전환 중에는 기존 모델과 새 모델이 잠시 함께 메모리를 사용한다. Jetson 장치에서 실제 모델과 CUDA/TensorRT의 전환을 확인해야 한다.

## 표시 이미지 생성의 불필요한 작업 정리

- 히트맵은 기존 Matplotlib 설정과 PNG 픽셀을 유지하며 `BytesIO`에서 OpenCV로 전달한다. 저장 폴더에 임시 `{name}_heatmap.png`를 쓰고 다시 읽어 삭제하는 왕복이 없어졌다. 즉시 읽고 버리는 메모리 중간 PNG는 무압축 PNG로 만들어 압축 연산도 생략한다. 원본 PNG, 최종 `{name}_resulf.jpg` 및 CSV 저장 방식은 유지한다.
- 이미 만든 336×336 RGB 이미지를 재사용하고 중복 resize·PIL 변환·NumPy 복사를 제거했다. 색상 범위, 투명도, tight bbox, 크롭과 최종 표시 크기는 유지한다.
- score < threshold인 OK에서는 최종적으로 사용하지 않던 윤곽선 탐색·색칠·확대를 생략한다. NG와 score == threshold의 윤곽선 처리 및 map >= threshold 기준은 유지한다.
- 전처리·모델 실행·점수·판정·카메라 동작, `AI_result_flag`의 추가 결과창과 최종 GUI·로그 처리 순서는 유지한다. 새 설정이나 의존성은 없다.

`tests/fixtures/mosa_rendering_2203676.py`에 변경 전 커밋의 렌더링 구간을 AST로 추출해 독립 비교 기준으로 보관했다. GUI 이미지 3개와 추가 결과창의 입력 RGB, 최종 합성 JPG 바이트를 OK/NG/동일 threshold·여러 맵·크롭 위치에서 비교하고 DPI·배경색·투명 PNG 설정도 확인한다. 실제 Jetson의 시간 절감량은 장치에서 별도로 측정한다.

검증(2026-10-19): `MPLCONFIGDIR=/tmp/jetson-cv-v03-mpl-cache /tmp/jetson-cv-v03-check/bin/python -B -m unittest discover -s tests -v`에서 **96개 통과, skip 없음**. 렌더링 픽셀·JPG 비교는 27개 맵/점수/크롭 조합과 별도 스타일 조건을 포함한다. `git diff --check` 통과. Mac CPU/Agg·로컬 임시 저장소에서 이전/이후를 번갈아 31회씩 실행한 렌더링 구간 중앙값은 OK 24.390→16.495ms, NG 24.214→17.748ms였다. 이 수치는 Jetson이나 전체 처리시간의 가속률을 뜻하지 않는다.

## USB CSV 로그의 반복 접근 줄이기

`CSV_path`의 `log.csv`를 GUI가 처음 기록할 때 열고 재사용한다. 같은 파일에 이어 쓸 때는 존재·크기 조회, 헤더 읽기와 파일 열기·닫기를 반복하지 않는다. 매 측정의 한 행은 즉시 추가하고 `flush()`하며, CSV 열·인코딩·줄바꿈과 구형 로그의 `backend=unknown` 변환은 유지한다. USB에 직접 저장하며 새 설정 키는 없다.

- `CSV_path`가 바뀌면 다음 로그 기록에서 기존 파일을 닫고 새 파일을 검증한 뒤 연다. `Data_log_flag=0`은 다음 측정의 설정 읽기에서 파일을 닫는다. GUI 정상 종료와 메인 종료 정리에서도 닫는다.
- 쓰기·flush 오류는 기존 측정 오류 창으로 알리고 열린 파일을 정리한다. 다음 측정에서는 파일을 다시 검증한다. 실패한 행은 일부 기록됐을 수 있으므로 자동으로 다시 추가하지 않는다.
- `flush()`는 Python 버퍼를 운영체제에 전달한다. 기존처럼 `fsync()`는 호출하지 않으며 물리 장치 기록 완료를 보장하지 않는다. USB 자체의 쓰기·동기화 지연은 남을 수 있다.

파일을 열어 둔 동안 같은 CSV를 외부에서 교체·삭제하는 것은 감시하지 않는다. USB 제거나 CSV 교체 전에는 GUI를 종료하거나 로그를 끄고 다음 측정으로 적용해 파일을 닫는다. 교체 파일은 `MOSA_visualAD_comb_jetson.py`와 `scripts/mosa_jetson_bridge.py`다. 모델·전후처리·판정·이미지 저장은 바꾸지 않았다.

검증(2026-10-19): `MPLCONFIGDIR=/tmp/jetson-cv-v03-mpl-cache /tmp/jetson-cv-v03-check/bin/python -B -m unittest discover -s tests -v`에서 **110개 통과, skip 없음**. CSV 바이트와 매 행 가시성, 연속 기록 시 파일 재접근 생략, 구형 로그 변환, 경로 전환, 로그 재활성화, USB 열기·쓰기·flush·종료 오류와 GUI 종료 정리를 확인했다. 이전 커밋과 AST 비교에서 측정 파이프라인은 CSV 호출 대상 외에 동일했다. `git diff --check` 통과. 실제 USB에서의 전체 시간 감소량은 장치 로그로 확인한다.

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
- v03의 점수는 Gaussian 적용 전 raw map의 상위 0.005% 평균이다(비율 0.00005, 336×336에서는 ceil로 6픽셀). fused 출력과 마지막 사용 가능 레이어의 토큰 출력 모두 같은 점수·필터·threshold 규칙을 사용한다. 이전 네 레이어 합산, fused 최댓값 및 TopK 0.005에서 확인한 결과 일치는 과거 검증이며 새 버전은 다시 비교해야 한다.
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

측정 log.csv의 마지막 열 backend에 실제 로드한 onnx/tensorrt를 기록한다. 기존 형식의 로그는 처음 기록할 때 backend 열을 추가하며 이전 행은 unknown으로 보존한다. 실행 중 data.json을 수정한 뒤 모델 변경 버튼으로 항목을 다시 선택하면 새 backend 설정이 적용된다.

## 영상 속성 재현

`examples/mosa_data_jetson_fields.example.json`의 `video_controls` 객체를 기존 data.json에 병합한다. 밝기는 기존 최상위 Brightness를 사용한다. fixed 모드에서 객체를 생략하면 추가 영상 속성을 설정하지 않는다. Windows 프로필은 기본값을 채운다. 최초 시작·설정 변경·재연결 복구 시 적용하며, fixed 모드에서는 매 측정에도 적용한다. 자동 화이트밸런스를 먼저 끈 뒤 색온도를 설정하고 모든 지정 항목을 readback 검증한다. 변경은 다음 측정부터 반영된다. fixed 모드에서 항목을 삭제하면 기존 장치 값을 되돌리지는 않는다.

AM7115MZT 사용자 제공 범위로 검증한다. 예제의 대비16/채도32/색조0/감마5/선명도0/색온도5800은 기존 Windows 코드의 요청값이다. 실제 Windows readback 및 이미지 동등성은 별도 검증한다. power_line_frequency=2는 현재 Jetson의 60Hz 설정을 유지하는 값이며 LED 제어가 아니다. ColorEnable/BacklightCompensation/Gain은 해당 장치 UVC 목록에 없어 보내지 않는다. focus_absolute도 inactive 상태이므로 이번 영상 속성 설정에 포함하지 않는다.
