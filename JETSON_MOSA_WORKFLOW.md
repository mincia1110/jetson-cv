# 클린 Jetson → MOSA 메인 실행 통합 워크플로

이 문서가 설치·실행 절차의 통합 진입점이다. 목표는 Jetson Orin Nano Super에서 Dino-Lite를 제어하고 `MOSA_visualAD_comb_jetson.py`를 ONNX CUDA로 실행하여 오프라인 운용하는 것이다. TensorRT 전환은 선택 사항이다. ONNX 속도는 사용자 요구에 충분했으며, 현재 우선순위는 Windows GUI와 anomaly score 및 카메라 조건 비교다. TRT 양자화는 후순위다. 진행 중 새 의존성·오류가 확인되면 해당 단계와 마지막 검증 표를 갱신한다.

**SD카드 OS/DATA 분리·Windows 공유 설정:** [14절의 실제 작업 기록](#sd-data-workflow)을 참조한다.

## 0. 기준 환경과 준비물

- Jetson Orin Nano Super 개발 키트, microSD, DP 모니터, USB 키보드·마우스.
- Dino-Lite AM7115MZT (USB `a168:0960`).
- 목표 소프트웨어: JetPack 6.2.1, Ubuntu 22.04, Python 3.10, CUDA 12.6. UEFI 펌웨어 버전과 OS L4T 버전은 별개다.
- 설치 중 인터넷 허용, 최종 사용 시 인터넷 차단.
- 기존 사내 MOSA의 `data.json`, `utils`와 그 전이 의존성, ONNX 모델. 이미지·모델·실제 설정은 사내에 둔다.
- 이 저장소만 clone해도 사내 파일까지 생기는 것은 아니다. Windows DNX64.dll은 이 포트에서 사용하지 않는다.

현재 실제 설정과 상세 진행 기록은 `local/` 및 로컬 전용 `PROGRESS.md`에 보관한다. 이 문서의 모델 경로·설정 예시는 실제 사내 값이 아니다.

## 1. OS 설치와 시스템 확인

1. Windows PC에서 [JetPack 6.2.1 전용 페이지](https://developer.nvidia.com/embedded/jetpack-sdk-621)의 **Orin Nano Developer Kit SD card image**를 받는다. 최신 버전 일반 링크에서 다른 JetPack을 받지 않는다.
2. balenaEtcher에서 이미지 ZIP 선택 → 대상 microSD 선택 → Flash → 검증 완료까지 기다린다. 기록 후 Windows가 포맷을 제안하면 취소한다.
3. Jetson 전원을 끈 상태에서 microSD, DP 모니터, USB 키보드·마우스를 연결하고 부팅한다. microSD는 부팅뿐 아니라 Ubuntu·프로젝트·패키지·결과 저장에도 사용한다. `df -h /`로 여유 공간을 확인한다.
4. JetPack 5.x용 구형 펌웨어 장치는 공식 초기 펌웨어 업데이트가 선행돼야 한다. 이번 사용자 장치에서는 UEFI 36.4.7을 확인했으며, 이미 JetPack 6.2.1로 구동 중인 보드에 펌웨어 업데이트를 반복할 필요는 없다.

Etcher에서 `h.requestMetadata` 계열 오류가 발생했던 이력이 있다. 구버전이 필요한 경우 [공식 Releases](https://github.com/balena-io/etcher/releases)에서 해당 Windows 설치 파일을 받는다. 1.18.11은 당시 검토한 우회 후보이며 필수 버전이나 검증 완료 버전으로 간주하지 않는다.

 사용자 생성·네트워크 연결을 완료한다. 이미지 기록은 카드 내용을 지우므로 대상 카드를 확인한다. 공식 [시작 안내](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit), [펌웨어 안내](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)를 따른다.

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
sudo apt-get install git wget
mkdir -p ~/Documents/MOSA
cd ~/Documents/MOSA
git clone https://github.com/mincia1110/jetson-cv.git
cd jetson-cv
bash scripts/setup_jetson.sh
```

이미 clone했다면 해당 폴더에서 `git pull origin main`만 한다. setup_jetson.sh는 nvidia-jetpack, venv, pip, 시스템 OpenCV·NumPy, Tk/ImageTk, uvcdynctrl, v4l-utils, usbutils, wget과 기본 추론 도구를 설치한다. 모델별 의존성을 모두 설치하는 스크립트는 아니다.

## 3. 가상환경

setup 스크립트가 `.venv`를 만든다. 아래 생성 명령은 아직 venv가 없을 때만 실행하고, 이미 있으면 활성화부터 진행한다.

```bash
test -d .venv || python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip check
```

`virtualenv` 패키지는 필요 없다. `--system-site-packages`는 JetPack TensorRT와 시스템 OpenCV 등을 공유하기 위한 옵션이다. 기존 venv를 활성화했다면 재생성하지 않는다. `sudo pip`를 사용하지 않는다. NumPy는 시스템 OpenCV와 호환하도록 1.x 범위를 유지한다.

```bash
python -c "import sys, cv2, numpy, tkinter; from PIL import ImageTk; print(sys.executable); print(cv2.__version__, numpy.__version__)"
/usr/local/cuda/bin/nvcc --version
```

## 4. ONNX Runtime GPU 설치

Python 3.10 / JetPack 6.x / CUDA 12.6 기준:

```bash
python -m pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 'onnxruntime-gpu==1.24.0' 'numpy<2'
python -c "import onnxruntime as o; print(o.__version__); print(o.get_available_providers())"
```

사용자 환경에서 ORT 1.24.0과 CUDAExecutionProvider를 확인했고 실제 0 입력 실행도 완료했다. 위 버전은 기존 사용자 환경에서 확인한 값이다. 해당 인덱스에서 Python 3.10/aarch64 wheel을 더 이상 제공하지 않으면 버전을 임의로 바꾸지 말고 기존 장비에서 보관한 wheel이나 호환 빌드를 확인한다. 재현 설치는 검증 후 wheel과 버전을 보관한다. 인덱스 내용은 바뀔 수 있으므로 설치가 되었다는 사실만으로 동일 빌드라고 간주하지 않는다. [NVIDIA의 JetPack 6.2.1 안내](https://forums.developer.nvidia.com/t/onnx-runtime-for-jetpack-6-2-1/359081).

`CUDAExecutionProvider`가 없으면 다음 단계 전에 해결한다. CPU provider가 목록에 함께 있다는 사실만으로 CPU 실행이라고 판단하지 않는다. 프로젝트 ONNX 실행부는 CUDA 요청 시 CPU fallback을 차단한다.

## 5. Torch·Torchvision와 cuDSS

기존 MOSA의 get_transform은 torchvision을, VisualAD 후처리는 CPU torch·scipy를 사용한다. 모델 본체의 GPU 실행은 ONNX CUDA 또는 TensorRT다.

```bash
python -m pip install 'torch==2.11.0' 'torchvision==0.26.0' 'numpy<2' --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
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

### 영상 속성 및 노출

기존 data.json에 다음 객체를 추가한다. 마지막 항목 뒤에는 쉼표를 넣지 않는다. 예제 파일 전체로 기존 모델·저장 설정을 덮어쓰지 않는다.

```json
"video_controls": {
  "white_balance_automatic": 0,
  "white_balance_temperature": 5800,
  "contrast": 16,
  "saturation": 32,
  "hue": 0,
  "gamma": 5,
  "sharpness": 0,
  "power_line_frequency": 2
}
```

자동 WB가 켜져 있으면 색온도는 inactive다. 코드는 AWB OFF 후 색온도를 설정하고 지정값을 다시 읽어 확인한다. 초기 시작·설정 변경·복구에서 적용한다. fixed 모드에서는 매 측정에도 적용하지만, MOSA Windows 프로필에서는 같은 설정을 다시 쓰지 않는다. 밝기는 최상위 Brightness 하나로 관리한다. 미노출 ColorEnable/BacklightCompensation/Gain 및 inactive focus는 설정하지 않는다. Windows의 요청값과 같아도 이미지 동등성은 별도 확인해야 한다.

ExposureTime은 다음 문자열 중 하나다: `1/1000s`, `1/500s`, `1/250s`, `1/125s`, `1/60s`, `1/30s`, `1/15s`, `1/8s`, `1/4s`, `1/2s`, `1s`, `2s`, `4s`, `8s`, `16s`. 중간값은 미지원이다. 이는 USB 명령표의 이름이며 모든 값의 실제 셔터 시간을 검증한 것은 아니다. DNX64 ExposureValue와의 변환식은 확인되지 않았다.

### 설정 반영·자동복구·로그

- `reset_flag_en: 1` 또는 `true`: MOSA에서 실패 시 FAIL(WAIT) → 닫기 → 1초 대기 → 재연결·초기값 적용 → READY → 수동 재측정. after는 진단용이며 해당 클릭에서 추론하지 않는다. READY는 영상 조건 통과가 아니다. 다음 클릭도 광량이 기준 밖이면 추론하지 않는다. 재초기화 자체 실패 시 FAIL 유지. 0은 자동복구 OFF다.
- [Windows 800 모드](WINDOWS_800_CHECK.md)의 설정 보존과 before/after/수동 재측정 비교 절차를 따른다. 새 메인과 scripts를 함께 갱신한다. 독립 카메라 진단 GUI는 기존 자동 재검사 방식을 유지한다.
- 카메라 설정·검사기준은 다음 측정 시 다시 읽는다. backend·모델·엔진 경로·device/fps는 앱 재시작 후 적용한다.
- `Data_log_flag: 1`일 때 CSV_path의 log.csv에 실제 실행 backend를 기록한다. 기존 로그는 backend 열을 추가하고 이전 행을 unknown으로 보존한다. CSV_path는 미리 존재하는 쓰기 가능한 Linux 폴더여야 한다.
- Windows와 비교할 때 먼저 동일 PNG/BMP로 점수를 비교하고, 이후 고정 조명·시료·거리에서 카메라 설정을 조정한다. 이미지는 사내에서만 비교해도 된다. 촬영 원본은 PNG로 저장하며 파일 입력은 PNG 우선, BMP 대체 읽기를 지원한다. 새 `Camera_or_png_flag`와 기존 `Camera_or_BMP_flag` 모두 사용 가능하며 새 키가 우선한다.
- Windows v03(2026-10-19 반영)에서는 fused/토큰 출력 모두 TopK ratio 0.00005(상위 0.005%, 336×336에서 6픽셀)를 사용한다. fused 맵은 필요시 336×336으로 보간하며, 토큰 모델은 마지막 사용 가능 레이어 하나를 쓴다. 점수 계산 후 Gaussian sigma=4로 맵을 필터링하고 label/mask는 사용자 threshold 초과 기준이다. GUI OK/NG는 score < threshold일 때 OK다. 이전 fused 최댓값·0 기준, 네 레이어 합산 및 0.005에서 확인한 결과 일치와 구분하여 재검증한다. 세부 변경과 교체 파일은 [포트 문서](MOSA_JETSON_PORT.md)를 참조한다.

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

모델 선택 → 폴더·이름·threshold 설정 → 측정 → 이미지/결과/CSV 저장 확인. 런타임 backend는 onnx로 둔다. 실제 utils가 추가 패키지를 요구하면 **import 오류에 나온 실제 의존성**을 확인해 설치하고 이 문서에 추가한다. 사용자는 기존 사내 자원을 이용한 메인 ONNX 실행 성공을 확인했다. 새 장비에서 같은 버전과 import 체인이 재현되는지는 이 단계에서 확인한다.

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
python scripts/doctor.py --output artifacts/environment/doctor.json
dpkg-query -W > artifacts/environment/dpkg.txt
git rev-parse HEAD > artifacts/environment/commit.txt
```

doctor.json의 error 항목도 확인한다. doctor는 환경 조회이며 GPU 실행 성공을 대신하지 않는다. 위 환경 기록 명령은 저장소 루트에서 실행한다. pip freeze만으로 JetPack·apt 공유 라이브러리는 복원되지 않으므로 dpkg 목록과 이미지 버전, 설치 wheel/DEB, 모델/엔진 해시도 함께 보관한다. 완전한 오프라인 신규 설치용 wheelhouse/apt 미러는 현재 자동화되지 않았다. 목표는 온라인 세팅 후 오프라인 운용이다.

## 진행 상태 — 2026-09-16

| 단계 | 확인 상태 |
| --- | --- |
| UVC 프레임 획득, LED 제어 | 사용자 실물 확인 |
| 고정 설정 재적용, 비정상 광량 차단 | 사용자 실물 확인 |
| ONNX Runtime 1.24.0 설치·CUDA 지정 0 입력 실행 | 사용자 실물 확인 |
| TensorRT FP32 엔진 실행 | 사용자 실물 확인 |
| raw outputs 수치 비교 | 일부 출력 기본 오차 기준 초과, 보류 |
| 공통 VisualAD 후처리 구현 | 로컬 테스트 완료, 사내 실제 결과 비교 필요 |
| cuDSS 검색 오류 | 패키지 설치 후 경로 누락 확인, 경로 지정으로 해결 사용자 확인. 영구 등록·새 셸 확인은 별도 |
| MOSA 메인 ONNX·TRT | 사용자 실행·성능 확인, ONNX 우선 운용 |
| 영상 속성 초기화·readback | 구현 및 로컬 테스트 완료, 실물 적용 확인 필요 |
| Windows GUI와 결과 일치 | 동일 BMP 점수 비교 → 동일 시료 촬영 조건 조정 예정 |
| 네트워크 차단 후 재부팅·최종 인수 | 아직 미확인 |

이 표는 증거가 생겼을 때 갱신한다. 사내 실제 설정·개별 로그와 자세한 일지는 로컬 PROGRESS.md에 기록한다.

## 자주 발생한 오류

| 증상 | 확인 및 조치 |
| --- | --- |
| virtualenv 없음 | 추가 설치 대신 `python3 -m venv --system-site-packages .venv` 사용 |
| JSON file load Failed | 작업 폴더의 data.json 경로, 마지막 쉼표, 중괄호 확인. 새 코드는 오류 경로와 줄·열 출력 |
| 고정 ExposureTime 선택 오류 | 실제 읽는 data.json에 정확한 시간 문자열 입력. null 또는 Windows ExposureValue로 대체 불가 |
| 설정을 바꿨는데 미반영 | 다른 JSON을 수정했는지 확인. 메인은 data.json만 읽음. backend는 재시작 필요 |
| mosa_jetson_bridge 모듈 없음 | 메인 파일만 복사하지 말고 저장소 scripts/까지 함께 유지 |
| libcudss.so.0 없음 | 5단계에서 설치 여부와 검색 경로를 각각 점검 |
| tqdm/ftfy/regex/sklearn/tabulate/skimage/seaborn 없음 | 같은 venv의 `python -m pip install -r requirements-inference.txt` 실행 |
| cv2/NumPy ABI 오류 | 시스템 OpenCV 사용, numpy<2 유지. pip OpenCV를 중복 설치하지 않음 |
| 자동복구 안 됨 | reset_flag_en 및 report.reset_succeeded 확인. READY는 재초기화 완료로, 수동 재측정 필요. 영상 조건 미달이 지속되면 다음 측정도 추론 차단 |

모든 설치가 끝나면 `python -m pip check`와 10단계 import 검사를 다시 수행한다. 이 문서는 확인된 작업 흐름이며, 새 SD에서 전체 절차를 처음부터 재실행한 인수 결과는 아직 없다.

<a id="sd-data-workflow"></a>
## 14. microSD OS/DATA 분리와 Windows 공유 — 2026-10-28

이 절은 **이미 설정된 Jetson의 OS를 보존하면서 같은 microSD에 공유 DATA 파티션을 추가한 실제 작업 기록**이다. 사내 Windows PC의 부팅 USB·상용 파티션 도구를 사용하지 않고, **Ubuntu 20.04 Linux NBPC의 GParted + exfat-utils**로 작업했다. 새 카드에 전체 설치를 재현한 결과와는 구분한다.

장비 작업의 확인 상태는 제공된 작업 기록을 옮긴 것이며, 이 패치 적용 세션에서 실물 작업을 재수행한 것은 아니다.

### 14.1 확인된 환경과 최종 구성

| 항목 | 이번 작업에서 확인한 값 |
| --- | --- |
| Jetson Linux / 커널 | `/etc/nv_tegra_release`: R36.4.7 / `5.15.148-tegra` |
| microSD | 256GB 제품, Linux 표시 약 238.3GiB, GPT |
| 기존 OS | APP, `/dev/mmcblk0p1`, ext4, 축소 전 사용량 약 36GiB |
| 최종 OS / DATA | APP 128GiB / exFAT 약 108.8GiB |
| DATA 이름·볼륨 레이블 | `JETSON_DATA` |
| Jetson 마운트 위치 | **`/mnt/jetson_data`** |
| Jetson 드라이버 | `exfat-fuse` 1.3.0+git20220115, `findmnt`에는 `fuseblk`로 표시 |
| DATA GPT 유형 | **Microsoft Basic Data**, `EBD0A0A2-B9E5-4433-87C0-68B6B72699C7` |

이때 확인한 **OS의 L4T R36.4.7**은 앞의 초기 설치 절차에서 기대하는 R36.4.4 및 UEFI 버전 기록과 별개다. 이 저장장치 작업을 이유로 `setup_jetson.sh`를 재실행하거나 버전 검사를 제거하지 않는다. 현재 장비의 JetPack 메타패키지 버전까지 새로 확인한 것은 아니다.

`lsblk`의 파티션 번호순을 물리적 위치순으로 해석하면 안 된다. 이번 카드의 `parted ... print free` 결과에서는 **p2~p15가 앞쪽 약 1~1493MiB, p1(APP)가 그 뒤부터 카드 끝까지**였다. 따라서 APP의 시작을 그대로 두고 끝만 축소했으며, 기존 부팅·복구 파티션은 이동하지 않았다.

```text
물리 배치(개념도)
[NVIDIA p2~p15 약 1.46GiB] [p1 APP ext4 128GiB] [p16 DATA exFAT 약 108.8GiB]
```

### 14.2 Jetson에서 exFAT 지원과 FUSE 확인

```bash
head -n 1 /etc/nv_tegra_release
uname -r
zcat /proc/config.gz | grep -E 'CONFIG_(EXFAT_FS|VFAT_FS|FUSE_FS)'
sudo modprobe fuse
lsmod | grep '^fuse'
```

실제 결과는 `CONFIG_FUSE_FS=m`, `CONFIG_VFAT_FS=y`, `# CONFIG_EXFAT_FS is not set`이었다. `modprobe exfat`은 모듈 없음으로 실패했지만 FUSE 모듈은 정상 로딩됐다. **exfatprogs는 포맷·검사 도구이지 누락된 커널 exFAT 드라이버를 설치하는 패키지가 아니다.**

먼저 설치 계획을 확인하고, 기존 패키지 제거가 없는 경우 설치한다.

```bash
sudo apt-get -s install exfat-fuse exfatprogs
sudo apt install exfat-fuse exfatprogs
command -v mount.exfat-fuse
command -v mkfs.exfat
ls -l /dev/fuse
```

이번 계획은 `exfat-fuse`, `exfatprogs`, `libfuse2` 신규 3개, 제거 0개였다. `autoremove` 안내는 별개이며 실행하지 않았다. 다른 장비에서 중요한 패키지 제거가 예고되면 설치를 중단하고 의존성을 확인한다.

실제 파티션 작업 전 128MiB 임시 이미지에 loop 장치를 연결하여 exFAT 포맷 → FUSE 마운트 → `Jetson exFAT OK` 기록 → 언마운트·재마운트 → 내용 읽기를 검증했다. 이후 loop 장치와 임시 이미지를 정리했다. 이는 기본 호환성 검증이며 장시간 쓰기 성능 시험은 아니다.

### 14.3 Ubuntu 20.04 NBPC에서 APP 축소·DATA 생성

**기존 카드 전체 이미지 또는 복구 가능한 백업을 먼저 확보한다. 파티션 축소는 데이터 손상 위험이 있으며, 백업 완료 여부를 이 기록만으로 가정하지 않는다.** Jetson을 정상 종료하고 SD카드를 NBPC에 연결한다. 실행 중인 Jetson 루트 파티션을 그 Jetson에서 축소하지 않는다.

```bash
# Ubuntu 20.04 NBPC에서 실행
sudo apt update
sudo apt install gparted e2fsprogs exfat-utils
lsblk -o NAME,PATH,SIZE,TYPE,FSTYPE,LABEL,PARTLABEL,MOUNTPOINT
sudo gparted
```

NBPC에서 `exfatprogs`를 찾지 못했던 문제는 Ubuntu 20.04의 `exfat-utils`를 사용해 해결했다. 이 환경에서는 **GParted로 unformatted 파티션 생성 후 터미널에서 exFAT 포맷**했다. 최신 GParted의 exFAT 생성 지원 여부를 이 구버전 환경에 그대로 적용하지 않는다.

GParted에서 약 238.3GiB **SD카드**를 선택한다. 내부 SSD와 혼동하지 않는다. 필요하면 자동 마운트된 SD 파티션을 해제한다.

| 작업 | 설정 |
| --- | --- |
| APP → Resize/Move | `Free space preceding` 기존 값 유지, `New size = 131072 MiB`, `Align to = MiB` |
| APP 축소 적용 | 시작 위치와 기존 p2~p15가 바뀌지 않는지 확인 후 적용 |
| 새로 생긴 큰 Unallocated 영역 → New | 남은 공간 최대 크기, Primary Partition, MiB 정렬 |
| 새 파티션 이름 / 파일시스템 | `Partition name = JETSON_DATA`, `File system = unformatted`, Label 공란 |
| 생성 후 Manage Flags | DATA에 **`msftdata`** 설정, 실제 GPT 유형도 별도 확인 |

APP 시작 위치 약 1493MiB는 조회 화면의 반올림 표시다. 그 숫자를 직접 재입력하지 말고 기존 시작 위치를 보존한다. 작업 중 전원을 끄거나 카드를 분리하지 않는다. `Create Partition Table`, APP 포맷·삭제, 기존 시스템 파티션 이동은 하지 않는다.

참고: [GParted 공식 매뉴얼](https://gparted.org/display-doc.php?name=help-manual), [GNU Parted의 msftdata 설명](https://www.gnu.org/software/parted/manual/html_node/set.html).

### 14.4 NBPC에서 새 DATA만 exFAT으로 포맷

**다음 `/dev/sdc16`은 이번 세션에서 사용자가 확인한 장치명이다. 다시 연결하거나 다른 PC에서 작업하면 바뀔 수 있다.** 매번 용량·파티션 이름·장치명을 확인하고, 아래 포맷 명령의 대상을 실제 DATA 파티션으로 바꾼다.

```bash
lsblk -o NAME,PATH,SIZE,FSTYPE,LABEL,PARTLABEL,MOUNTPOINT /dev/sdc
```

이번에는 `/dev/sdc16`이 약 109GiB의 새 `JETSON_DATA`임을 확인했다. 마운트되어 있으면 해당 파티션만 해제한 뒤 포맷했다.

```bash
# 파괴적 명령: 확인한 새 DATA 파티션에만 실행한다.
sudo mkfs.exfat -n JETSON_DATA /dev/sdc16
sudo blkid /dev/sdc16
lsblk -o NAME,SIZE,FSTYPE,LABEL,PARTLABEL,PARTTYPE /dev/sdc
sync
```

Ubuntu 20.04 `exfat-utils`의 레이블 옵션은 **`-n`**, Jetson의 `exfatprogs`에서는 **`-L`**이다. 서로 바꿔 쓰지 않는다. `TYPE="exfat"`와 `LABEL="JETSON_DATA"`를 확인한다. **포맷과 GPT 유형 지정은 별개이므로 PARTTYPE도 확인해야 한다.** 기존 `/dev/sdc1` 또는 전체 디스크 `/dev/sdc`를 포맷 대상으로 지정하지 않는다. 카드의 모든 마운트를 해제하고 안전하게 분리한다.

옵션 근거: [Ubuntu 20.04 mkexfatfs 매뉴얼](https://manpages.ubuntu.com/manpages/focal/man8/mkexfatfs.8.html).

### 14.5 Jetson 수동 마운트와 `/data` 충돌 해결

Jetson의 파일 관리자에서는 네이티브 exFAT 미지원 오류가 발생했다. 재포맷하지 않고 이미 검증한 FUSE 구현을 명시했다.

처음 시도한 `/data`는 비어 있지 않아 `fuse: mountpoint is not empty`로 실패했다. 확인 결과 기존 **`/data/tee`**가 있었다. 이 경로를 삭제·이동하거나 `nonempty`로 가리지 않고, 별도 마운트 지점을 선택했다.

```bash
lsblk -o NAME,SIZE,FSTYPE,LABEL,PARTLABEL,MOUNTPOINT
sudo mkdir -p /mnt/jetson_data
sudo mount.exfat-fuse \
  -o uid=$(id -u),gid=$(id -g),umask=022 \
  /dev/disk/by-label/JETSON_DATA /mnt/jetson_data
findmnt --mountpoint /mnt/jetson_data
```

마운트 성공을 확인한 뒤에만 저장 테스트를 실행한다.

```bash
if mountpoint -q /mnt/jetson_data; then
    printf 'Jetson exFAT OK\n' > /mnt/jetson_data/test.txt
    sync
    cat /mnt/jetson_data/test.txt
else
    echo 'ERROR: DATA is not mounted; do not write here.' >&2
fi
```

실제 SD 파티션에서 `Jetson exFAT OK` 읽기·쓰기에 성공했다. `fuseblk` 표시는 정상이며 네이티브 커널 exFAT이 활성화됐다는 의미는 아니다.

### 14.6 UUID 기반 자동 마운트

```bash
sudo blkid /dev/disk/by-label/JETSON_DATA
id -u
id -g
sudo cp -a /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d-%H%M%S)"
# nano가 없으면 sudo apt install nano
sudo nano /etc/fstab
```

다음은 **예제**다. `ACTUAL-UUID`와 `1000` 두 값을 실제 UUID·UID·GID로 치환한다. 기존 동일 마운트 지점 행이 있으면 수정하고 중복 추가하지 않는다.

```fstab
UUID=ACTUAL-UUID /mnt/jetson_data exfat-fuse rw,nofail,uid=1000,gid=1000,umask=022,x-systemd.device-timeout=10s 0 0
```

네이티브 지원이 없는 이 장비에서는 `exfat` 대신 **`exfat-fuse`**를 사용한다. `nofail`은 DATA 마운트 실패가 OS 부팅을 막지 않게 하기 위한 옵션이지, 데이터 저장이 가능하다는 보장이 아니다.

MOSA와 해당 경로를 사용하는 파일 관리자·터미널을 닫고 검증한다.

```bash
cd "$HOME"
sync
sudo umount /mnt/jetson_data
sudo systemctl daemon-reload
sudo mount /mnt/jetson_data
findmnt --mountpoint /mnt/jetson_data
cat /mnt/jetson_data/test.txt
```

오류가 있으면 재부팅 전에 fstab을 수정한다. 성공 후 `sudo reboot`하고 `findmnt --mountpoint /mnt/jetson_data`로 재확인했다. **사용자가 재부팅 후 자동 마운트 성공까지 확인했다.**

### 14.7 Windows에서 안 보였던 원인과 실제 해결

Windows 디스크 관리에는 약 109GB 파티션이 보였지만 파일시스템이 공란이었고 탐색기에 나타나지 않았다. PowerShell 조회에서 DATA의 GPT 유형이 아래 Linux Filesystem GUID로 남아 있는 것을 확인했다.

```text
변경 전: 0FC63DAF-8483-4772-8E79-3D69D8477DE4  (Linux Filesystem)
변경 후: EBD0A0A2-B9E5-4433-87C0-68B6B72699C7  (Microsoft Basic Data)
```

관리자 PowerShell에서 조회한다. `N`은 확인한 SD카드 디스크 번호다.

```powershell
Get-Disk | Format-Table Number,FriendlyName,Size,PartitionStyle
Get-Partition -DiskNumber N |
  Format-Table PartitionNumber,Size,GptType,DriveLetter,IsHidden,NoDefaultDriveLetter
```

DiskPart에서 **SD카드 번호 N과 DATA 번호 P를 매번 확인**한다. 이번 DATA는 16번이었지만 번호를 무조건 가정하지 않는다.

```text
diskpart
list disk
select disk N
list partition
select partition P
detail partition
```

약 109GB DATA이고 현재 유형이 위 Linux GUID임을 확인한 경우에만 실행한다.

```text
set id=ebd0a0a2-b9e5-4433-87c0-68b6b72699c7
detail partition
exit
```

이번에는 `override` 없이 변경에 성공했고, **Windows 탐색기에 DATA와 내부 파일이 정상 표시됨을 사용자가 확인했다.** 포맷·파티션 재생성 없이 GPT 유형만 수정했다. 드라이브 문자가 자동 할당되지 않으면 디스크 관리에서 해당 DATA에만 빈 문자를 할당한다.

`clean`, 디스크 초기화, Linux/부팅 파티션 포맷·삭제는 금지한다. Windows가 다른 파티션의 포맷을 제안하면 취소한다. 이 수정은 DATA의 유형을 바꾸는 것이며 APP나 EFI 유형을 바꾸는 절차가 아니다. [Microsoft DiskPart set id 문서](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/set-id)의 대상 선택 경고도 참고한다.

### 14.8 MOSA 저장 경로와 미마운트 방어

마운트 확인 후에만 폴더를 만든다. 사용자 UID/GID로 마운트했으므로 일반 사용자로 생성한다.

```bash
if mountpoint -q /mnt/jetson_data; then
    mkdir -p /mnt/jetson_data/inference_results /mnt/jetson_data/logs
else
    echo 'ERROR: DATA is not mounted; no directories were created.' >&2
fi
```

MOSA GUI의 **저장 폴더**에서 `/mnt/jetson_data/inference_results`를 선택한다. 기존 `data.json`에서는 다음 항목만 실제 운용에 맞게 수정하고 나머지 설정을 유지한다.

```json
{
  "CSV_path": "/mnt/jetson_data/logs",
  "Data_log_flag": 1
}
```

위 JSON은 기존 파일에 병합할 두 필드의 예시이며 전체 data.json 대체 파일이 아니다. 이미지 폴더는 GUI 선택값, CSV 폴더는 별도 설정이다. 코드·모델·venv는 기존 ext4에 유지한다.

`MOSA_visualAD_comb_jetson.py`의 `_require_data_mount()`는 **`/mnt/jetson_data` 자체와 하위 경로**를 대상으로 `os.path.ismount()`를 확인한다. 단순히 폴더가 존재하는지 확인하는 것으로 대체하지 않는다. 다음 시점에서 검사한다.

- GUI 폴더 선택, CSV 로깅이 활성화된 설정 로드(시작·측정 시).
- 측정 시작 전과 비동기 카메라 준비 완료 후, 원본 이미지 쓰기 전.
- 추론 후 결과 JPG 쓰기 직전과 CSV append 직전.

미마운트이면 명확한 오류를 내고 해당 측정·저장을 중단한다. 자동 마운트·sudo·디렉터리 생성·OS 파티션으로의 대체 저장은 수행하지 않는다. DATA를 가리키는 심볼릭 링크도 검사하며, DATA 아래 경로가 링크를 통해 밖으로 빠져나가면 거부한다. 심볼릭 링크 뒤의 `..`를 먼저 제거하면 실제 경로와 달라질 수 있으므로 실제 경로 해석을 먼저 수행한다. 결과 JPG의 `cv2.imwrite()` 실패도 오류 처리한다.

**기존 로컬 테스트 호환성을 위해 명시적으로 선택한 DATA 밖 경로는 계속 허용한다.** 이 검사는 저장 위치를 자동 변경하거나 모든 OS 저장을 금지하는 정책이 아니다. Windows 공유가 목적이면 GUI 폴더와 CSV_path를 모두 위 경로로 지정해야 한다. `run_mosa_logged.py`의 별도 터미널 로그 등 다른 스크립트까지 이번 보호 범위를 확장한 것은 아니다.

이는 마운트 누락·측정 사이 언마운트를 탐지하는 방어다. 검사와 쓰기 사이의 강제 언마운트 경쟁, 잘못된 장치의 동일 위치 마운트, 매체 불량·공간 부족까지 보장하지 않는다. 앱 실행 중 언마운트하지 않고, 오류 발생 시 이미 일부 저장된 파일이 있을 수 있음을 확인한다. [Python ismount 문서](https://docs.python.org/3/library/os.path.html#os.path.ismount) 참고.

검증 명령(모델·카메라·디스플레이 없이 실행):

```bash
python3 -m unittest discover -s tests -p test_mosa_storage.py -v
python3 -m py_compile MOSA_visualAD_comb_jetson.py tests/test_mosa_storage.py
```

### 14.9 확인 상태와 운용 수칙

| 항목 | 확인 상태 |
| --- | --- |
| 128MiB loop 이미지 포맷·재마운트·파일 유지 | 사용자 Jetson 확인 |
| APP 축소 및 실제 DATA exFAT 생성 | 사용자 확인 |
| 실제 DATA FUSE 읽기·쓰기 | 사용자 확인 |
| fstab 설정·재부팅 후 자동 마운트 | 사용자 확인 |
| GPT 유형 수정 후 Windows에서 DATA·내부 파일 보기 | 사용자 확인 |
| Windows에서 새 파일 작성 후 Jetson 재읽기 | 아직 확인 보고 없음 |
| 추가한 MOSA 마운트 방어 | 로컬 단위 테스트 20개 통과, 수정 코드의 Jetson GUI 실물 검증은 별도 |
| 장시간 연속 추론 저장 성능·매체 수명 | 별도 검증 필요 |

Jetson에서 MOSA를 종료하여 CSV 등 열린 파일을 닫고 `sync` 후 `sudo poweroff`한다. 전원이 완전히 꺼진 다음 SD카드를 옮긴다. OS 카드이므로 실행 중 빼지 않는다. Windows에서도 안전하게 제거한다. exFAT 공유 볼륨은 결과 교환용이며 유일한 백업으로 사용하지 않는다.
