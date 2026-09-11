# 클린 Jetson → MOSA 메인 실행 통합 워크플로

이 문서가 설치·실행 절차의 통합 진입점이다. 목표는 Jetson Orin Nano Super에서 Dino-Lite를 제어하고 `MOSA_visualAD_comb_jetson.py`를 ONNX CUDA로 실행한 뒤 TensorRT로 전환하여 오프라인 운용하는 것이다. 진행 중 새 의존성·오류가 확인되면 해당 단계와 마지막 검증 표를 갱신한다.

## 0. 기준 환경과 준비물

- Jetson Orin Nano Super 개발 키트, microSD, DP 모니터, USB 키보드·마우스.
- Dino-Lite AM7115MZT (USB `a168:0960`).
- 목표 소프트웨어: JetPack 6.2.1, Ubuntu 22.04, Python 3.10, CUDA 12.6. UEFI 펌웨어 버전과 OS L4T 버전은 별개다.
- 설치 중 인터넷 허용, 최종 사용 시 인터넷 차단.
- 기존 사내 MOSA의 `data.json`, `utils`와 그 전이 의존성, ONNX 모델. 이미지·모델·실제 설정은 사내에 둔다.
- 이 저장소만 clone해도 사내 파일까지 생기는 것은 아니다. Windows DNX64.dll은 이 포트에서 사용하지 않는다.

현재 실제 설정과 상세 진행 기록은 `local/` 및 로컬 전용 `PROGRESS.md`에 보관한다. 이 문서의 모델 경로·설정 예시는 실제 사내 값이 아니다.

## 1. OS 설치와 시스템 확인

Windows에서 JetPack 6.2.1 Orin Nano 개발 키트용 SD 이미지를 받아 microSD에 기록하고 Jetson을 부팅한다. 사용자 생성·네트워크 연결을 완료한다. 이미지 기록은 카드 내용을 지우므로 대상 카드를 확인한다. 공식 [시작 안내](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit), [펌웨어 안내](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)를 따른다.

```bash
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core
uname -m
python3 --version
```

예상: Ubuntu 22.04, aarch64, Python 3.10. 프로젝트 설치 스크립트는 L4T `36.4.4-*`를 요구한다. UEFI 36.4.7이라고 해서 nvidia-l4t-core도 36.4.7인 것은 아니다. 다른 OS면 설치기 검사만 삭제하지 말고 실제 호환성을 확인한다.

## 2. 저장소·JetPack·기본 패키지

```bash
sudo apt-get update
sudo apt-get install git
mkdir -p ~/Documents/MOSA
cd ~/Documents/MOSA
git clone https://github.com/mincia1110/jetson-cv.git
cd jetson-cv
bash scripts/setup_jetson.sh
sudo apt-get install python3-tk python3-pil.imagetk uvcdynctrl
```

이미 clone했다면 해당 폴더에서 `git pull origin main`만 한다. setup_jetson.sh는 nvidia-jetpack, venv, pip, 시스템 OpenCV·NumPy, v4l-utils, usbutils와 기본 추론 도구를 설치한다. 모델별 의존성을 모두 설치하는 스크립트는 아니다.

## 3. 가상환경

setup 스크립트가 `.venv`를 만든다. 수동 생성할 때는 다음 명령을 사용한다.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip install 'matplotlib<3.9' 'numpy<2'
```

`virtualenv` 패키지는 필요 없다. `--system-site-packages`는 JetPack TensorRT와 시스템 OpenCV 등을 공유하기 위한 옵션이다. 기존 venv를 활성화했다면 재생성하지 않는다. `sudo pip`를 사용하지 않는다. NumPy는 시스템 OpenCV와 호환하도록 1.x 범위를 유지한다.

```bash
python -c "import sys, cv2, numpy, tkinter; from PIL import ImageTk; print(sys.executable); print(cv2.__version__, numpy.__version__)"
/usr/local/cuda/bin/nvcc --version
```

## 4. ONNX Runtime GPU 설치

Python 3.10 / JetPack 6.x / CUDA 12.6 기준:

```bash
python -m pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu 'numpy<2'
python -c "import onnxruntime as o; print(o.__version__); print(o.get_available_providers())"
```

사용자 환경에서 ORT 1.24.0과 CUDAExecutionProvider를 확인했고 실제 0 입력 실행도 완료했다. 재현 설치는 검증 후 wheel과 버전을 보관한다. 인덱스 내용은 바뀔 수 있으므로 설치가 되었다는 사실만으로 동일 빌드라고 간주하지 않는다. [NVIDIA의 JetPack 6.2.1 안내](https://forums.developer.nvidia.com/t/onnx-runtime-for-jetpack-6-2-1/359081).

`CUDAExecutionProvider`가 없으면 다음 단계 전에 해결한다. CPU provider가 목록에 함께 있다는 사실만으로 CPU 실행이라고 판단하지 않는다. 프로젝트 ONNX 실행부는 CUDA 요청 시 CPU fallback을 차단한다.

## 5. Torch·Torchvision와 cuDSS

기존 MOSA의 get_transform은 torchvision을, VisualAD 후처리는 CPU torch·scipy를 사용한다. 모델 본체의 GPU 실행은 ONNX CUDA 또는 TensorRT다.

```bash
python -m pip install torch torchvision 'numpy<2' --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

### `ImportError: libcudss.so.0`일 때

이번 사용자 환경에서 실제 발생한 오류다. cuDSS 0.6.0.5-1 설치 후에도 검색 경로 누락으로 지속되었으며, 아래 경로 지정으로 해결됐음을 사용자가 확인했다. 당시 torch 2.11.0 / torchvision 0.26.0이었다.

```bash
mkdir -p artifacts/installers
cd artifacts/installers
wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudss0-cuda-12
sudo ldconfig
cd ../..
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
```

[동일 환경 설치 사례](https://forums.developer.nvidia.com/t/help-me-with-correct-pytorch-and-torchvision-versions-requirement-for-jetpack-6-2-1-orin-super/343688?page=2). 다른 공유 라이브러리 오류가 나오면 그 이름과 torch 버전을 기록하고 별도로 해결한다. 무관한 .so를 이름만 바꿔 연결하지 않는다.

### 설치 후에도 같은 오류가 지속될 때 — 확인된 경로 문제

`dpkg -L libcudss0-cuda-12`에 파일이 있지만 `ldconfig -p`에 없었던 사례다. 현재 터미널에서 다음을 적용해 해결을 확인했다.

```bash
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/libcudss/12${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
```

재로그인에도 적용하려면 시스템 검색 경로를 등록한다. 사용자 보고로 확인된 것은 경로 문제 해결이며, 아래 영구 등록·새 셸 검증 완료 여부는 별도다.

```bash
echo '/usr/lib/aarch64-linux-gnu/libcudss/12' | sudo tee /etc/ld.so.conf.d/mosa-cudss.conf
sudo ldconfig
ldconfig -p | grep cudss
```

새 터미널에서 venv를 활성화하고 import를 다시 확인한다. 이 결과는 cuDSS 검색 문제 해결이며 전체 MOSA 실행 또는 Torch CUDA 연산 검증을 대체하지 않는다.

### 기존 VisualAD import 체인의 추가 패키지

MOSA 실행 중 `tqdm`, `ftfy`, `regex`, `sklearn`, `tabulate`, `skimage`, `seaborn` 누락이 차례로 보고되었고 사용자가 설치했다. requirements-inference.txt에 반영했으므로 신규 환경은 3단계 설치에 포함된다.

```bash
python -m pip install tqdm ftfy regex scikit-learn tabulate scikit-image seaborn
python -c "import tqdm, ftfy, regex, sklearn, tabulate, skimage, seaborn; print('VisualAD helper imports OK')"
```

설치 패키지 이름은 `scikit-learn`, Python import 이름은 `sklearn`이다. `scikit-image`의 import 이름은 `skimage`다. 개별 설치 성공과 전체 메인 실행 성공은 구분한다. 정확한 설치 버전은 최종 pip-freeze 기록에 보관한다.

## 6. 카메라 설정·검증

```bash
lsusb
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-ctrls-menus
mkdir -p local artifacts
cp camera_check.example.json local/camera_check.json
```

이미 실제 설정 파일이 있으면 복사로 덮어쓰지 않는다. JSON에는 검증한 Brightness, ExposureTime, BRIGHT_min/max, RG_gab, capture_no를 입력한다. USB 번호는 재연결 시 바뀔 수 있으므로 장치를 확인한다.

```bash
python scripts/camera_control_gui.py --device /dev/video0 --config local/camera_check.json
```

- null ExposureTime은 아직 고정 노출이 선택되지 않은 상태다. 드롭다운에서 선택하고 파일에도 저장한다.
- 시작 시 출력되는 **실제 JSON 절대 경로**를 확인한다. 다른 JSON을 수정해서 검사 기준이 반영되지 않은 사례가 있었다.
- AE ON 후 조건 검사에서 초기 고정값 복원, 비정상 광량 유지 시 재검사 후 추론 차단을 확인한다.
- reset=false는 고정값 미적용이 아니라 재연결 불필요라는 뜻이다.
- 노출 readback은 없음. 시간 명령 전송을 실제 셔터 조회값이라고 해석하지 않는다. Windows ExposureValue 숫자는 자동 변환하지 않는다.
- 이 검증 화면은 AI를 실행하지 않는다. 하단 10줄의 R 평균·R−G로 광량을 판정한다.

상세는 CAMERA_CONTROL.md. 최종 MOSA 메인은 첨부본처럼 **전체 프레임 평균화**를 사용하며 이전 카메라 검증 화면의 ROI 평균화와 다르다.

## 7. 사내 MOSA 자원 배치

두 폴더를 구분한다.

- `jetson-cv`: Git 저장소, 새 메인·연결 모듈·venv·엔진·local 설정.
- 기존 MOSA 폴더: 사내 data.json, utils/transforms.py, VisualAD_lib 등 실제 전처리 의존성, ONNX.

기존 data.json의 모델 A~E 경로, CSV_path, 파일 저장 관련 값은 Linux에서 유효하게 수정한다. 현재 메인은 기존 키들을 계속 읽으므로 Subspace 비활성화와 무관하게 sub_shift/sub_scale 등 기존 키는 유지한다. 윈도우 드라이브 경로를 그대로 사용하지 않는다.

## 8. 런타임 설정

`examples/mosa_data_jetson_fields.example.json`은 기존 data.json에 추가할 항목 예제다. 기존 파일 전체를 덮어쓰지 말고 병합한다. 이전 jetson_runtime.json의 backend/provider/input_name/device/fps를 data.json 최상위에 옮긴다.

- backend: 먼저 `onnx`, 이후 `tensorrt`.
- provider: `CUDAExecutionProvider`.
- 카메라 설정: 6단계에서 검증한 camera_check.json의 Brightness, ExposureTime, BRIGHT_min, BRIGHT_max, RG_gab, capture_no, reset_flag_en, settle_frames를 기존 MOSA data.json으로 옮긴다. 모델·저장·GUI 키는 보존한다. runtime의 camera_config는 더 이상 사용하지 않는다. 메인 검증 후 독립 진단 GUI에서 사용하지 않는 camera_check.json은 삭제 가능하다. ExposureValue는 실제 노출 명령에 사용하지 않는다.
- 모델별 엔진: model_A_path_engine ~ model_E_path_engine. 기존 model_A_path_vis ~ model_E_path_vis와 같은 항목에 대응한다. 이전 engines 매핑의 경로를 각 키에 옮긴다.
- `input_name`: 현재 실제 모델은 `input`.

MOSA 메인은 실행 작업 폴더의 **data.json만** 읽는다. MOSA_RUNTIME_CONFIG 환경변수와 jetson_runtime.json은 더 이상 사용하지 않으므로 이관 후 제거할 수 있다. 상대 모델·엔진 경로도 실행 작업 폴더 기준이다.

## 9. ONNX 단독 검사

저장소 루트에서 실제 모델 절대 경로로 실행한다. 출력 폴더는 매번 새 이름을 사용한다.

```bash
python scripts/onnx_probe.py --model /absolute/path/model.onnx --output artifacts/onnx-io-01
python - <<'PY'
import numpy as np
np.savez('artifacts/onnx-smoke-input.npz', input=np.zeros((1,3,336,336),dtype=np.float32))
PY
python scripts/onnx_probe.py --model /absolute/path/model.onnx --inputs artifacts/onnx-smoke-input.npz --output artifacts/onnx-run-01
```

현재 모델은 입력 [1,3,336,336], 특징 3개 [1,1024], 레이어 6/12/18/24 토큰 [1,579,1024]를 출력한다. 출력은 최종 맵이 아니며 visualad_adapter가 후처리한다. DRM vendor 탐색 경고가 있어도 실제 세션·추론이 성공한 사례가 있다. 경고만 보고 성공/실패를 단정하지 않는다.

## 10. MOSA 메인 ONNX 실행

아래 `/absolute/path`는 실제 절대 경로로 바꾼다. **실행 디렉터리는 data.json이 있는 기존 MOSA 폴더**다.

```bash
cd /absolute/path/original/MOSA
source /absolute/path/jetson-cv/.venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, torchvision, scipy, matplotlib, cv2, onnxruntime; from utils.transforms import get_transform; print('imports OK')"
python /absolute/path/jetson-cv/MOSA_visualAD_comb_jetson.py
```

모델 선택 → 폴더·이름·threshold 설정 → 측정 → 이미지/결과/CSV 저장 확인. 런타임 backend는 onnx로 둔다. 실제 utils가 추가 패키지를 요구하면 **import 오류에 나온 실제 의존성**을 확인해 설치하고 이 문서에 추가한다. 아직 전체 사내 import 체인은 검증되지 않았다.

## 11. TensorRT 변환·수치 비교

대상 Jetson에서 저장소 루트로 돌아와 실행한다.

```bash
bash scripts/build_engine.sh /absolute/path/model.onnx artifacts/visualad-fp32
python scripts/compare_backends.py --onnx /absolute/path/model.onnx --engine artifacts/visualad-fp32/model.engine --inputs artifacts/onnx-smoke-input.npz --output artifacts/backend-comparison-01
python scripts/compare_visualad.py --reference artifacts/backend-comparison-01/onnx_outputs.npz --actual artifacts/backend-comparison-01/trt_outputs.npz --threshold 0 --output artifacts/visualad-postprocess-01
```

threshold 0은 예시다. 실제 GUI 입력값을 /10한 내부 threshold를 사용한다. FP32부터 비교하고 임의로 허용오차를 늘려 통과 처리하지 않는다. 이미 0 입력에서 일부 특징 출력이 기본 허용오차를 넘었다. 최종 score/map·경계 판정 영향은 추가 검증해야 한다. 실제 BMP·입출력 NPZ는 사내에 두고 요약 오차·판정만 공유 가능 범위에서 기록한다.

## 12. TensorRT 메인 실행

data.json을 backend=tensorrt로 바꾸고 모델별 model_A_path_engine ~ model_E_path_engine을 채운다. 10단계와 같은 명령으로 메인을 실행한다. ONNX와 동일한 사내 전처리 및 CPU 후처리를 사용한다. Subspace 추론 주석과 함수는 유지하며 시작 시 미사용 모델 로드는 비활성화했다.

같은 사내 BMP/촬영 조건에서 ONNX와 score·map·OK/NG·처리시간을 비교한다. 이 포트는 기존 동기 GUI 흐름을 유지하므로 측정 중 잠시 화면이 멈출 수 있다. DLL의 모든 영상 속성 초기화까지 이식된 것은 아니므로 색상·감마 동일성은 별도 확인한다.

## 13. 오프라인 인수와 재현 자료

설치·모델·엔진 준비 후 인터넷을 끊고 재부팅한다. 10단계 환경 설정을 다시 적용한 뒤 모델 선택부터 촬영·판정·저장·종료·재시작을 확인한다. 네트워크 다운로드가 실행 경로에 필요하면 완료로 보지 않는다.

```bash
mkdir -p artifacts/environment
python -m pip freeze > artifacts/environment/pip-freeze.txt
python -m pip check
python scripts/doctor.py --help
dpkg-query -W > artifacts/environment/dpkg.txt
git rev-parse HEAD > artifacts/environment/commit.txt
```

위 환경 기록 명령은 저장소 루트에서 실행한다. pip freeze만으로 JetPack·apt 공유 라이브러리는 복원되지 않으므로 dpkg 목록과 이미지 버전, 설치 wheel/DEB, 모델/엔진 해시도 함께 보관한다. 완전한 오프라인 신규 설치용 wheelhouse/apt 미러는 현재 자동화되지 않았다. 목표는 온라인 세팅 후 오프라인 운용이다.

## 진행 상태 — 2026-09-11

| 단계 | 확인 상태 |
| --- | --- |
| UVC 프레임 획득, LED 제어 | 사용자 실물 확인 |
| 고정 설정 재적용, 비정상 광량 차단 | 사용자 실물 확인 |
| ONNX Runtime 1.24.0 설치·CUDA 지정 0 입력 실행 | 사용자 실물 확인 |
| TensorRT FP32 엔진 실행 | 사용자 실물 확인 |
| raw outputs 수치 비교 | 일부 출력 기본 오차 기준 초과, 보류 |
| 공통 VisualAD 후처리 구현 | 로컬 테스트 완료, 사내 실제 결과 비교 필요 |
| cuDSS 검색 오류 | 패키지 설치 후 경로 누락 확인, 경로 지정으로 해결 사용자 확인. 영구 등록·새 셸 확인은 별도 |
| MOSA 메인 실행·ONNX 추론 | 사용자 실물 성공 확인 (설정 통합 전); 통합 후 재확인 필요 |
| 네트워크 차단 후 재부팅·최종 인수 | 아직 미확인 |

이 표는 증거가 생겼을 때 갱신한다. 사내 실제 설정·개별 로그와 자세한 일지는 로컬 PROGRESS.md에 기록한다.
