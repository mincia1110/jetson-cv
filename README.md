> 클린 Jetson부터 MOSA 메인 실행까지는 [통합 워크플로](JETSON_MOSA_WORKFLOW.md)를 먼저 읽으세요.

# Jetson VisualAD 실행환경

Orin Nano Super Developer Kit + microSD + DP 모니터 + USB 키보드/마우스용 준비 키트.
설치는 온라인, 운영은 오프라인. ONNX는 변환 입력이다. 대상 Jetson에서 TensorRT 엔진(.engine)을 만들고, 기존 Python/PyQt 앱은 촬영 → cv2 전처리 → TensorRT GPU 추론 → 결과 표시로 연결한다.
실제 장비와 사내 모델은 아직 연결되지 않았으므로 이 폴더는 **환경 준비 자료이며 실기 검증 완료본이 아니다.**

## 따라가는 순서와 완료 기준

처음에는 0 → 1 → 2 → 3 → 4의 합성 모델 검사까지 진행한다. 이후 사내 모델과 기존 앱을 확보해 4의 실제 모델 변환 → 5 → 6으로 진행한다. 카메라가 인식되지 않으면 3의 증거를 남기고 4의 파일 입력 검사로 넘어갈 수 있다.

| 단계 | 실행 위치 | 성공 확인 |
| --- | --- | --- |
| 0. 자료 받기 | 인터넷 가능한 Windows PC | README와 scripts 폴더 확인 |
| 1. OS 설치 | Windows + Jetson | Jetson에서 Ubuntu 데스크톱 부팅 |
| 2. 환경 설치 | 온라인 Jetson | 설치 종료 코드 0, cv2/numpy/tensorrt/polygraphy import 성공 |
| 3. 카메라 | Jetson | 저장한 PNG가 실제 대상 영상인지 확인 |
| 4. 합성 모델 | Jetson | 엔진 생성, report.json의 pass=true와 comparison.pass=true |
| 4–5. 실제 모델 | 사내 Jetson | 실제 엔진 생성과 기준 입력/출력 비교 성공 |
| 5. 기존 앱 연결 | 사내 Jetson | 촬영 버튼으로 결과 표시, 오류 없이 반복 촬영 |
| 6. 오프라인 | 네트워크를 끊은 Jetson | 재부팅 후 같은 흐름 성공 |

**명령 예시 규칙:** `/path/...`, 입력 이름 `input`, `336x336`은 실제 값으로 바꾼다. 정적/동적 모델 명령은 해당하는 하나를 선택한다. Python 앱 연결 예시는 기존 함수에 맞춰 편집하는 코드이며 그대로 실행하는 완성 앱이 아니다. 새 터미널에서는 프로젝트 폴더로 이동하고 `source .venv/bin/activate`를 다시 실행한다. 검사 재실행 시 `artifacts/smoke-build-02`처럼 새 출력 경로를 사용한다.

## 0. 프라이빗 저장소 받기와 사내 반입

GitHub에 로그인한 Windows 브라우저에서 [저장소](https://github.com/mincia1110/jetson-cv)를 열고 **Code → Download ZIP**으로 받는다. 압축을 풀면 README와 scripts가 같은 폴더에 있어야 한다. 이 폴더를 사내 반입 절차에 따라 USB 등으로 전달한다. Git이나 GitHub 인증 도구를 Jetson에 설치할 필요는 없다.

Git을 사용하는 경우 인터넷 가능한 PC에서 `git clone https://github.com/mincia1110/jetson-cv.git`으로 받는다. 토큰을 URL에 넣지 말고 기존 GitHub 인증을 사용한다. 버전을 고정한 소스 ZIP은 해당 체크아웃에서 `git archive --format=zip --output=../jetson-cv-source.zip HEAD`로 만들 수 있다. `git rev-parse HEAD` 결과도 별도로 기록한다.

**소스 ZIP에는 JetPack 이미지, 설치 패키지, 실제 모델, 사내 앱이 포함되지 않는다.** 모델/입출력 기준 NPZ/카메라 영상은 사내에서 별도 관리한다. 설치 단계에는 인터넷이 필요하며, 소스 ZIP만으로 처음부터 오프라인 설치가 가능한 것은 아니다.

## 기준 환경

| 항목 | 기준 |
| --- | --- |
| OS | JetPack 6.2.1 / L4T 36.4.4 / Ubuntu 22.04 / ARM64 |
| GPU 구성 | 해당 JetPack의 CUDA 12.6 / cuDNN 9.3 / TensorRT 10.3 |
| Python | 시스템 Python 3.10 + system-site-packages venv |
| 영상 입력 | Dino-Lite가 UVC 지원 시 V4L2 / OpenCV |
| 화면 | 사내 PyQt 앱 재사용, PyQt 주 버전은 사내 확인 후 설치 |
| GPU 기본 점검 | NVIDIA trtexec + 작은 합성 ONNX |
| 앱 추론 엔진 | TensorRT 10.3 엔진 직접 실행, NVIDIA Polygraphy TrtRunner 재사용 |

6.2.1은 이 키트의 고정 기준이며 최신 전체 JetPack이라는 의미가 아니다. 현재 공식 빠른 시작 문서는 7.x ISO 경로를 안내하므로 아래 **6.2.1 전용 링크**를 사용한다. 사내 의존성이 이 기준과 충돌하면 설치 전에 기준을 조정한다.

## 1. Windows에서 OS 준비

1. microSD 카드 리더, microSD(64GB 이상), DP 모니터, 키보드/마우스, 보드 전원과 냉각팬을 준비한다. 모델과 설치 파일 공간은 별도로 확보한다.
2. 전원을 켜고 NVIDIA 화면에서 Esc를 눌러 UEFI 버전을 기록한다. 구형 펌웨어라면 [공식 6.x 펌웨어 업데이트 절차](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)를 먼저 수행한다. 이 절차는 Windows PC와 microSD로 진행할 수 있다. 펌웨어 업데이트 중 전원을 끄지 않는다.
3. [JetPack 6.2.1 페이지](https://developer.nvidia.com/embedded/jetpack-sdk-621)에서 Orin Nano용 SD 카드 이미지를 받는다. Windows의 Balena Etcher로 대상 **microSD의 용량/드라이브를 확인하고** 기록·검증한다. 기록 시 카드의 기존 데이터는 지워진다. 다운로드한 이미지와 해시를 보관한다.
4. 카드를 Jetson에 넣고 DP/USB/네트워크를 연결한다. 초기 Ubuntu 계정을 만들고, 예약된 펌웨어 업데이트가 있으면 공식 절차대로 재부팅한다.
5. 이 폴더를 USB 등으로 Jetson에 복사한다. 아래 명령은 **Jetson의 이 폴더에서** 실행한다.

## 2. 온라인 환경 설치

```bash
bash scripts/setup_jetson.sh
source .venv/bin/activate
python scripts/doctor.py --output artifacts/doctor-online.json
python -c "import cv2, numpy, tensorrt, polygraphy; print('Required imports OK; TensorRT', tensorrt.__version__)"
```

설치기는 OS/아키텍처/L4T가 기준과 다르면 중단한다. nvidia-jetpack, 시스템 OpenCV, v4l-utils, venv와 합성 모델 생성용 onnx==1.16.2와 NVIDIA 실행 도우미 polygraphy==0.49.9를 설치한다. Windows DLL, PyTorch 학습 환경, 새로운 GUI는 설치하지 않는다. venv에서 시스템 OpenCV와 TensorRT를 사용한다. pip의 opencv-python으로 덮어쓰지 않는다.

기존 앱이 PyQt5라면 `sudo apt install python3-pyqt5`, PyQt6라면 해당 앱 버전과 ARM64 지원을 확인해 별도로 설치한다. 현재는 어느 버전인지 확인되지 않았다.

`doctor.py`는 현황 수집 도구다. 종료 코드 0만으로 환경 준비 완료로 판단하지 않는다. 필수 모듈의 error와 trtexec 경로를 확인한다. 사용하지 않는 PyQt 주 버전의 import 오류는 무시할 수 있다. 필수 import가 실패하면 4로 진행하기 전에 설치 로그를 확인한다.

## 3. 카메라를 모델과 별도로 점검

```bash
lsusb
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --list-ctrls-menus
python scripts/capture.py --device /dev/video0 --output artifacts/capture.png
```

`/dev/video0`은 예시다. 확인한 장치 또는 `/dev/v4l/by-id/...` 경로를 지정한다. 미리보기에서 Space로 PNG 한 장 저장, Q/Esc로 취소한다. 화면 없이 점검하려면 `--headless`를 쓴다. 필요할 때만 `--width`, `--height`, `--fourcc MJPG`를 지정하며 실제 협상값이 출력된다.

접근 거부라면 장치 권한과 `id`를 확인한다. video 그룹이 필요한 경우 `sudo usermod -aG video "$USER"` 후 로그아웃/로그인한다. chmod 777은 사용하지 않는다. 장치가 없거나 영상 획득에 실패하면 모델명·USB VID/PID와 출력 내용을 기록하고, 파일 입력 경로로 GPU 검증을 계속한다. Windows SDK 비호환만으로 UVC 불가라고 판단하지 않는다. LED/노출 등은 기존 검사 조건과 비교해 나중에 검증한다.

기존 앱에는 `cv2.VideoCapture(device, cv2.CAP_V4L2)`를 사용하는 작은 변경부터 적용한다. 기존 BGR/RGB 처리, crop, resize, normalization, dtype, 배치/축 순서는 유지한다. 이 키트의 촬영 창은 장치 진단용이며 기존 PyQt 화면을 대체하지 않는다.

## 4. ONNX → TensorRT 엔진 생성

변환은 **대상 Jetson에서 사전 수행**하고 운영 앱에는 검증된 엔진을 전달한다. NVIDIA `trtexec`를 사용하며 빌드와 추론을 분리한다. 입력 shape와 전처리를 모르는 상태에서 336×336 등으로 고정하지 않는다.

```bash
python scripts/make_smoke_model.py --output artifacts/smoke.onnx
bash scripts/build_engine.sh artifacts/smoke.onnx artifacts/smoke-build
python scripts/trt_probe.py artifacts/smoke-build/model.engine --inputs artifacts/smoke.inputs.npz --expected artifacts/smoke.expected.npz --output artifacts/smoke-result
```

작은 MatMul 모델을 변환하고 저장된 엔진을 다시 로드해 실제 입력/출력을 비교한다. `build.log`, 빌드 명령, ONNX/엔진 해시, `outputs.npz`, `report.json`을 남긴다. 출력 디렉터리는 새 경로를 사용한다. 합성 모델 성공은 VisualAD 모델의 변환·정확도·메모리 적합성 보장이 아니다.

실제 모델은 사내에서 아래 절차를 수행한다.

```bash
# 정적 shape 모델: 기본은 FP32 기준 비교를 위해 TF32 비활성화
bash scripts/build_engine.sh /path/model.onnx artifacts/visualad-fp32

# 동적 shape 모델 예시: 이름과 크기를 실제 모델에 맞게 변경
bash scripts/build_engine.sh /path/model.onnx artifacts/visualad-dynamic \
  --minShapes=input:1x3x336x336 \
  --optShapes=input:1x3x336x336 \
  --maxShapes=input:1x3x336x336
```

동적 입력은 모든 입력에 대해 min/opt/max를 지정한다. 수동 단일 촬영은 실제 전처리 해상도에서 batch=1을 우선 사용한다. shape tensor 입력이 있다면 NVIDIA의 해당 모델 프로파일 설정을 확인한다. 이 키트는 프로파일 0을 사용한다.

FP32 기준 결과를 확인한 뒤 별도 출력 경로로 `--fp16` 빌드를 수행하고 같은 샘플로 비교한다. FP16 플래그는 내부 연산의 저정밀 사용을 허용하며 입력 배열을 무조건 float16으로 바꾸라는 뜻이 아니다. INT8은 현재 범위에 포함하지 않는다.

변환 실패 시 `build.log`에서 미지원 연산, 동적 shape, 플러그인, 메모리 부족을 확인한다. 앱을 CPU로 전환하여 성공 처리하지 않는다. 필요한 커스텀 플러그인은 ARM64/TensorRT 버전에 맞춰 별도 준비하고 빌드와 런타임에서 로드해야 하며, 현재 범용 래퍼에는 사용자 플러그인 로더가 없다. 실제 모델에 필요하면 확장한다.

엔진은 GPU/플랫폼/TensorRT 버전의 영향을 받으므로 Windows에서 만든 엔진을 그대로 배포하지 않는다. JetPack/TensorRT/모델/shape/정밀도가 바뀌면 대상에서 재빌드·재검증한다. ONNX external-data 파일도 변환 시 함께 있어야 한다. ViT 기반 실제 모델은 보드의 메모리 안에서 빌드/실행되는지 별도로 측정한다.

## 5. 실제 입력 비교와 기존 PyQt 앱 연결

GPU 메모리·CUDA stream 관리 코드를 새로 작성하지 않고 NVIDIA의 [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)를 사용한다. `scripts/trt_session.py`는 저장된 엔진을 로드하는 얇은 어댑터다. ONNX Runtime과 CPU fallback은 사용하지 않는다. 런타임 의존성 자동 설치를 비활성화하며 누락 시 실패한다.

사내 기존 전처리 직후의 배열을 **실제 입력 이름을 키로** NPZ에 저장하고, 검증된 원본 모델의 출력을 같은 방식으로 저장한다. RGB/BGR, resize/crop, normalization, dtype, 축 순서, 후처리를 유지한다.

```python
np.savez("sample.inputs.npz", **feed)
np.savez("sample.expected.npz", **reference_outputs)  # 출력 이름 -> 배열
```

```bash
python scripts/trt_probe.py artifacts/visualad-fp32/model.engine --inputs /path/sample.inputs.npz --expected /path/sample.expected.npz --output artifacts/visualad-check
```

실행 또는 비교 실패는 종료 코드 1과 보고서를 남긴다. `--expected`를 생략하면 실행만 검사하며 `comparison: null`이다. 기본 허용오차는 rtol=1e-4, atol=1e-5이며 `--rtol`, `--atol`로 변경한다. 실제 정상/이상 샘플에 대해 허용오차와 최종 점수/판정/히트맵을 사내 기준으로 검증한다. 1회 측정 시간에는 데이터 전송 등이 포함되므로 GPU 커널 벤치마크로 해석하지 않는다.

기존 PyQt 앱 추론 worker에 아래 구조를 적용한다. 예시의 함수들은 기존 앱 함수로 연결해야 한다.

```python
from scripts.trt_session import TensorRTSession

# 같은 worker 스레드에서 생성/실행/종료. 매 촬영마다 엔진을 로드하지 않는다.
with TensorRTSession("artifacts/visualad-fp32/model.engine") as session:
    while running:
        frame = wait_for_capture()
        feed = existing_preprocess(frame)  # {실제 입력 이름: numpy 배열}
        outputs = session.infer(feed)
        result = existing_postprocess(outputs)
        result_ready.emit(result)  # Qt signal로 GUI 스레드에 전달
```

어댑터는 실제 입력 이름/shape/dtype를 검사하고 출력 배열을 복사해 다음 촬영 시 덮어쓰이지 않도록 한다. 한 세션을 여러 스레드에서 동시에 사용하지 않는다. 저장된 TensorRT 엔진을 역직렬화하고 추론하는 경로이며 실행 시 ONNX 변환은 없다. 기존 사내 앱 파일이 확보되면 추론 호출부만 실제 코드에 맞춰 연결한다.

기존 `trt_smoke.sh`는 랜덤 입력 벤치마크용으로 남아 있다. 변환과 정확도 검증의 기본 경로는 위 `build_engine.sh` + `trt_probe.py`이다.

## 6. 오프라인 인수 및 재현 기록

온라인 설치·사내 앱 연결이 끝난 뒤:

```bash
python scripts/doctor.py --output artifacts/doctor-final.json
python -m pip freeze --all > artifacts/pip-freeze.txt
dpkg-query -W > artifacts/dpkg-packages.txt
python -m pip check
```

다운로드 이미지, 직접 설치한 wheel/의존성, 앱 소스와 커밋, ONNX 및 external-data 파일, TensorRT 엔진/빌드 명령/플러그인, 전처리 설정, 기준 입력/출력, 임계값, 폰트/아이콘/Qt 플러그인, 로그를 사내 저장소에 보관한다. 해시는 `sha256sum <파일>`로 기록한다. pip freeze와 dpkg 목록만으로 오프라인 재설치를 보장하지 않는다. 재설치용으로는 검증된 microSD를 전원 종료 후 전체 이미지로 백업하고 별도 카드 복원까지 확인한다. venv만 다른 OS로 복사하지 않는다.

- [ ] 카메라 모델/VID/PID, 사용 해상도·색상·노출 조건 기록
- [ ] 사내 PyQt/TensorRT/Polygraphy 버전과 실행 명령 확정
- [ ] 모델/설정/기준 데이터 해시 기록, 실제 TensorRT 엔진 실행·기준 출력 비교 증거 확보
- [ ] 실제 앱에서 촬영 → 기존 전처리 → GPU 추론 → 결과 표시 성공
- [ ] 유선 연결을 빼고 Wi-Fi를 끈 후 재부팅
- [ ] 기존 앱을 수동 실행하고 촬영·추론·표시를 다시 수행
- [ ] 최초 실행/빈 사용자 캐시에서도 필요한 파일이 로컬에 있어 다운로드 요청 없이 실행
- [ ] 파일 입력 비교와 실제 촬영 검사를 구분하여 결과 기록
- [ ] 카메라 연결/해제, 앱 종료/재시작, 반복 촬영 시 동작 확인
- [ ] 실패 시 로그를 남기며 CPU만으로 실행된 것을 성공으로 표시하지 않음

오프라인 여부는 이 스크립트가 자동으로 네트워크를 끊어 판단하지 않는다. 실제 단절 상태의 앱 실행이 최종 증거다. 카메라 호환성 실패 시 파일 입력 GPU 검증까지 완료하고 카메라 연동은 보류한다.

## 출처와 범위

공식 자료 확인일: 2026-09-05. [VisualAD](https://github.com/7HHHHH/VisualAD)는 연구 코드 참고용이며 사내 ONNX/GUI와 동일하다고 가정하지 않는다. [Dino-Lite SDK 안내](https://www.dinolite.us/support/about-the-dino-lite-sdk/)와 [Linux UVC](https://www.ideasonboard.org/uvc/faq/)를 구분하여 확인한다. 표의 JetPack 구성은 [6.2.1 공식 페이지](https://developer.nvidia.com/embedded/jetpack-sdk-621)를 기준으로 한다.

## 기존 test.py의 ONNX/TensorRT 연결

측정 GUI 진입점은 `scripts/test_jetson.py`다. 카메라 교체, ONNX Runtime 실행, TensorRT 변환·비교·GUI 전환 절차는 [INFERENCE_PORT.md](INFERENCE_PORT.md)를 따른다. 실제 모델의 get_transform/user_th_inference 연결과 Jetson GPU 검증은 아직 필요하다.

첨부 MOSA 메인 기반 최소 변경판은 `MOSA_visualAD_comb_jetson.py`이며, 실행 절차는 [MOSA_JETSON_PORT.md](MOSA_JETSON_PORT.md)를 따른다.
