# Jetson 신규 셋업 및 MOSA 검사 GUI 포팅 결과

- 작성일: 2026-09-21
- 코드 기준: `0cdecc6` (본 문서 작성 직전 main)
- 대상: Jetson Orin Nano Super / Dino-Lite AM7115MZT / MOSA VisualAD GUI
- 근거: 개발 대화의 사용자 실측 보고, Windows GUI 로그·USB 캡처, 저장소 구현 및 테스트 기록
- 용도: 사내 개발 보고서 작성 참고. 상세 설치 실행서는 [JETSON_MOSA_WORKFLOW.md](JETSON_MOSA_WORKFLOW.md)를 참조한다.

## 1. 추진 목적 및 요약

기존 Windows PC에서 실행하던 MOSA 검사 GUI를 Jetson으로 이전해, 카메라 제어부터 영상 전처리·AI 추론·결과 표시·저장까지 장치 내에서 수행하는 환경을 구축했다. 새 GUI를 전면 개발하는 대신 기존 MOSA 메인의 구조와 작업 흐름을 유지하고 Windows 종속 카메라 제어와 추론 실행부를 교체했다.

ONNX Runtime CUDA와 TensorRT 실행 경로를 구현했으며, 사용자는 두 방식의 실행과 성능을 확인했다. ONNX 속도가 현재 요구에 충분해 우선 운용 경로로 선정했고 TensorRT 양자화는 후순위로 두었다. 동일 BMP에서 Windows와 Jetson 추론 결과가 일치함을 사용자가 확인했으며, 이후에는 촬영 영상의 색감·광량을 맞추는 작업에 집중했다.

카메라 색감 차이는 Windows 리셋 USB 명령을 분석하고 추가 화이트밸런스 쓰기를 단계적으로 제거하면서 개선됐다. `no_added_awb` 조건에서 리셋 후 수동 재측정의 경향이 일관적이라는 사용자 확인을 받았다. USB 물리적 재연결 후에는 장치 재탐색과 조건부 AWB 해제를 추가했고, 후속 로그에서 해당 경로와 설정 readback의 정상 진행을 확인했다. 이 경로의 장기간 색감·점수 안정성까지 확인된 것은 아니다.

## 2. 환경 구성

| 항목 | 신규 구축 기준 / 관측 환경 |
|---|---|
| 하드웨어 | Jetson Orin Nano Super, 시스템 표시 RAM 약 7.4 GiB |
| 설치 매체 | JetPack 6.2.1 Orin Nano용 이미지를 microSD에 기록하여 부팅 |
| OS / 아키텍처 | Ubuntu 22.04 / ARM64(aarch64) |
| 초기 L4T 기준 | 36.4.4 |
| 개발 중 실제 변경 | 2026-09-17 시스템 업데이트로 L4T 36.4.7 적용 |
| 실행 커널 | 사용자 출력 기준 5.15.148-tegra |
| Python | 3.10, `.venv` + `--system-site-packages` |
| GPU 소프트웨어 기준 | CUDA 12.6, TensorRT 10.3 계열 |
| ONNX Runtime | 사용자 설치·실행 확인 버전 1.24.0 (`onnxruntime-gpu`) |
| Torch / Torchvision | 사용자 설치 확인 버전 2.11.0 / 0.26.0 |
| NumPy | 사용자 설치 확인 버전 1.26.4 |
| cuDSS | 사용자 설치 확인 버전 0.6.0.5-1 |
| 카메라 | Dino-Lite AM7115MZT, USB VID:PID `a168:0960` |
| 최종 영상 모드 | 2592×1944, MJPEG, 10 fps 요청 |
| GUI | 기존 Tkinter / Pillow / Matplotlib 구조 유지 |

JetPack 메타패키지 버전, L4T 패키지 버전, UEFI 버전, `uname -r`은 각각 기록해야 한다. `5.15.148-tegra`라는 커널 문자열만으로 L4T 세부 빌드가 동일하다고 판단할 수 없다.

현재 `scripts/setup_jetson.sh`는 초기 기준인 L4T 36.4.4만 허용한다. 업데이트된 36.4.7 장비에서는 검사에 걸리므로, 이미 구축된 환경에 신규 설치 스크립트를 무조건 재실행하지 않는다. 업데이트 전후 패키지 목록과 검증 결과를 별도로 보관하는 것이 재현 설치의 기준이다.

## 3. 신규 Jetson 구축 절차

### 3.1 OS 및 기본 도구

1. JetPack 6.2.1 전용 Orin Nano SD 이미지를 준비한다.
2. PC의 이미지 기록 도구로 microSD에 기록·검증한다.
3. Jetson에서 초기 사용자·네트워크를 설정하고 OS/L4T/아키텍처를 확인한다.
4. 저장소와 사내 MOSA 리소스를 배치한다.
5. JetPack 개발 패키지와 Python·영상 제어 도구를 설치한다.

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core nvidia-jetpack
uname -r
uname -m
python3 --version
```

기본 시스템 의존성은 `nvidia-jetpack`, `python3-venv`, `python3-pip`, `python3-opencv`, `python3-numpy`, `python3-tk`, `python3-pil.imagetk`, `uvcdynctrl`, `v4l-utils`, `usbutils`다. 초기 기준 환경에서는 `bash scripts/setup_jetson.sh`로 기본 구성을 진행한다.

### 3.2 Python 환경

```bash
# 저장소 루트; .venv가 없을 때만 생성
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip check
```

`virtualenv` 별도 설치는 필요하지 않다. 시스템 OpenCV와 JetPack TensorRT를 공유하기 위해 system-site-packages를 사용한다. pip의 OpenCV wheel을 추가로 설치해 시스템 OpenCV와 혼용하지 않는다.

| 분류 | 구성 및 역할 |
|---|---|
| 기본 영상 | NumPy 1.x, Pillow, 시스템 OpenCV, Tk/ImageTk |
| 추론 도구 | ONNX, Polygraphy, 별도 설치한 ONNX Runtime GPU, JetPack TensorRT |
| 전처리·후처리 | Torch, Torchvision, SciPy |
| GUI·시각화 | Matplotlib, Pillow |
| 기존 VisualAD import 의존성 | tqdm, ftfy, regex, scikit-learn, tabulate, scikit-image, seaborn |

Torch·Torchvision·ONNX Runtime GPU는 ARM64/Python/CUDA 조합에 맞는 빌드를 별도로 설치했다. `requirements-inference.txt`만으로 모든 GPU 패키지와 사내 코드까지 설치되는 것은 아니다. 실제 사용한 wheel과 `pip freeze`를 보관해야 재설치 재현성을 높일 수 있다. 생성한 wheel-backup의 복구 절차와 한계는 3.5절에 정리했다. 본 보고서의 버전은 해당 장비에서 확인한 이력이며, 현재 온라인 인덱스의 배포 여부를 보증하는 목록은 아니다.

### 3.3 cuDSS 경로 문제 해결

Torch import 시 `ImportError: libcudss.so.0`가 발생했다. 패키지는 설치돼 있었지만 동적 라이브러리 검색 경로가 누락된 사례였으며, 아래 경로 지정으로 해결됐음을 사용자가 확인했다.

```bash
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/libcudss/12${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
```

영구 경로 등록 및 새 로그인 후 확인 방법은 통합 워크플로에 있다. 경로 문제 해결, import 성공, 실제 추론 성공은 각각 별도 확인 단계로 관리한다.

### 3.4 사내 리소스와 실행 위치

실제 ONNX 모델, 기존 `data.json`, `utils.transforms` 등 사내 전처리 코드는 별도로 필요하다. 공개 저장소를 받는 것만으로 전체 검사 자원이 갖춰지는 것은 아니다.

메인은 **현재 작업 폴더의 data.json**을 읽으며 모델·CSV 등의 상대 경로도 그 폴더를 기준으로 해석한다. Windows 드라이브 경로를 Linux 경로로 수정하고 메인과 `scripts/`를 함께 갱신한다.

### 3.5 생성한 wheel-backup을 이용한 복구

사용자가 기존 장비에서 안내한 백업 절차를 수행하고 `wheel-backup`을 생성했다고 보고했다. 백업 내용과 해시는 아직 이 작업 환경에서 직접 검사하지 않았으며, 신규 SD에서의 복구 성공도 별도 검증 대상이다.

#### 백업 구성과 복구 범위

```text
wheel-backup/
├── wheels/                  # 캐시에서 복사한 wheel + 다시 다운로드한 GPU wheel
├── gpu-requirements.txt     # 설치 시점 Torch/Torchvision/ORT GPU 버전
├── environment-venv.txt     # venv 로컬 패키지의 freeze --local --all
├── environment-all.txt      # 공유 시스템 패키지를 포함한 freeze --all
├── system-packages.txt      # dpkg-query -W 결과
├── python-version.txt
├── kernel.txt
├── cached-wheels.txt        # 기존 장비의 캐시 경로 기록; 복원 경로가 아님
└── SHA256SUMS
```

GPU 패키지는 `--no-deps`로 다운로드했으므로 **현재 백업만으로 전체 의존성이 갖춰졌다고 간주하지 않는다.** 캐시의 wheel도 현재 프로젝트와 무관한 파일을 포함할 수 있다. `wheels/*.whl` 전체를 한 번에 설치하지 않고 대상 패키지와 버전을 지정한다.

| 복구 대상 | 백업의 역할 | 별도로 필요한 것 |
|---|---|---|
| Torch/Torchvision/ONNX Runtime GPU | 보관한 호환 wheel로 재설치 | 해당 wheel이 요구하는 의존성 및 CUDA/cuDNN/cuDSS |
| 나머지 venv 패키지 | 버전 목록과 일부 wheel | 누락 wheel, 직접 URL·로컬 소스 설치 항목 |
| OS·JetPack·시스템 OpenCV·TensorRT | system-packages.txt로 비교 | SD 이미지와 시스템 패키지 설치 파일/저장소 |
| MOSA 소스·설정·모델 | wheel 백업에 포함되지 않음 | Git 커밋, data.json, 사내 utils, ONNX/엔진 별도 백업 |

`system-packages.txt`는 시스템 설치 파일 묶음이 아니며 이를 그대로 pip에 입력하지 않는다. `environment-all.txt` 역시 apt 관리 Python 패키지까지 pip로 덮어쓰는 복구 목록으로 사용하지 않는다. 재다운로드한 wheel의 SHA256은 보관 파일의 무결성을 확인하지만, 기존 장비가 최초 설치했던 wheel과 동일한 빌드임을 소급 증명하지 않는다.

#### A. 백업 파일과 대상 시스템 확인

백업 폴더 전체를 다른 저장 매체에도 보관한다. 새 장비에서 다음 명령을 **wheel-backup 폴더 안에서** 실행한다.

```bash
sha256sum -c SHA256SUMS
cat python-version.txt
cat kernel.txt
cat gpu-requirements.txt
```

해시 누락·불일치가 있으면 해당 파일을 다시 확보한다. SHA256SUMS에는 wheel 파일만 들어 있으며 환경 기록·설정·모델을 검증하는 목록은 아니다.

새 SD에 3.1절의 시스템 환경을 구성한 뒤 실제 버전을 비교한다.

```bash
uname -m
python3 --version
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core nvidia-jetpack libcudss0-cuda-12
```

ARM64/Python 3.10 기준을 맞추고 CUDA·cuDNN·TensorRT·cuDSS를 준비한다. 기존 장비는 L4T 36.4.7로 업데이트됐으므로 초기 36.4.4 SD 상태를 그대로 동일 환경이라고 부르지 않는다. 커널 이름만으로 버전을 판단하지 않고 system-packages.txt와 실제 패키지 버전을 대조한다. 서로 다르면 그 차이를 기록하고 호환성 검증을 수행한다.

#### B. 기존 운영 venv를 보존하고 복구용 venv 생성

아래는 저장소 루트에서 실행한다. 백업을 `artifacts/wheel-backup`에 복사했다고 가정한다. 다른 위치라면 MOSA_WHEEL_BACKUP만 실제 경로로 바꾼다. `.venv-restore`는 없는 이름이어야 하며 운영 중인 `.venv`에 덮어쓰지 않는다.

```bash
export MOSA_WHEEL_BACKUP="$PWD/artifacts/wheel-backup"
python3 -m venv --system-site-packages .venv-restore
source .venv-restore/bin/activate
python -c "import sys; print(sys.executable)"
```

시스템 OpenCV·TensorRT 공유가 목적이다. 복구 venv 경로는 로컬 자료로 관리하고 Git에 추가하지 않는다. cuDSS 경로가 필요한 장비에서는 3.3절의 LD_LIBRARY_PATH를 적용한다.

#### C. 보관 GPU wheel 설치

먼저 보관 폴더만 사용하는 설치를 시도한다.

```bash
python -m pip install \
  --no-index \
  --find-links="$MOSA_WHEEL_BACKUP/wheels" \
  --only-binary=:all: \
  -r "$MOSA_WHEEL_BACKUP/gpu-requirements.txt"
```

복구 때는 `--no-deps`를 사용하지 않는다. 의존성이 시스템 또는 venv에 없고 백업에도 없으면 설치가 실패할 수 있으며, 이는 불완전한 백업 범위를 드러내는 결과다. 버전을 임의로 바꾸거나 --no-deps로 무시하지 않는다. 패키지명·요구 버전을 기록하고 동일 환경의 온라인 장비에서 누락 wheel을 확보한 뒤 다시 실행한다. 다운로드에 사용한 인덱스/빌드 출처도 함께 기록한다.

`--no-index`는 패키지 인덱스 탐색을 막는다. 다만 requirements 안에 직접 HTTP URL이 있으면 그 URL 사용까지 막는 옵션은 아니므로, 오프라인용 목록에는 직접 URL이 없어야 한다. gpu-requirements.txt는 앞서 생성한 `이름==버전` 세 줄인지 확인한다. [pip 로컬 패키지 설치 안내](https://pip.pypa.io/en/stable/user_guide/#installing-from-local-packages)

#### D. 나머지 Python 의존성 복구

`environment-venv.txt`를 검토해 별도 `restore-requirements.txt`를 만든다. 일반적인 `이름==버전` 항목은 유지하고, `-e`, `@ file:`, HTTP URL, 로컬 프로젝트 항목은 보관한 소스/wheel을 사용하도록 개별 정리한다. GPU 세 패키지도 원래 버전으로 유지한다. 시스템 OpenCV·TensorRT를 다른 pip 배포판으로 치환하지 않는다.

모든 필요한 wheel이 준비되면 다음으로 복구한다.

```bash
python -m pip install \
  --no-index \
  --find-links="$MOSA_WHEEL_BACKUP/wheels" \
  --only-binary=:all: \
  -r "$MOSA_WHEEL_BACKUP/restore-requirements.txt"
python -m pip check
```

현재 캐시에는 의존성 전체가 없을 수 있어 위 명령의 성공을 미리 보장하지 않는다. 추가 다운로드 시 기존 GPU wheel은 보존하고, 일반 패키지와 Jetson 전용 패키지의 출처를 구분한다. 보충한 파일까지 포함하도록 백업 폴더에서 `sha256sum wheels/*.whl > SHA256SUMS`를 다시 생성한다.

인터넷을 사용하는 임시 복구에서는 3.2절의 requirements-inference.txt도 참고할 수 있지만, 범위 지정으로 원래보다 새 버전이 설치될 수 있다. 이러한 환경은 원본 버전과 대조·검증하기 전까지 동일 환경 복구로 보고하지 않는다.

#### E. import → 실제 추론 → GUI 검증

```bash
python -m pip check
python -c "import cv2, numpy, torch, torchvision, scipy, matplotlib, onnxruntime, tensorrt; print('imports OK'); print('torch:', torch.__version__, 'torchvision:', torchvision.__version__); print('ORT:', onnxruntime.__version__, onnxruntime.get_available_providers()); print('CUDA:', torch.cuda.is_available())"
python -c "import tqdm, ftfy, regex, sklearn, tabulate, skimage, seaborn; from PIL import ImageTk; import tkinter; print('GUI/helper imports OK')"
python -m pip freeze --local --all > "$MOSA_WHEEL_BACKUP/restored-environment-venv.txt"
diff -u "$MOSA_WHEEL_BACKUP/environment-venv.txt" "$MOSA_WHEEL_BACKUP/restored-environment-venv.txt"
```

차이가 있으면 각 패키지와 설치 출처 차이를 확인한다. pip check와 import 성공만으로 전체 GUI 복구 완료로 간주하지 않는다.

1. 백업한 Git 커밋의 소스, data.json, 사내 전처리 코드, 모델을 복원한다.
2. 기존 저장 입력으로 onnx_probe.py 실제 추론을 실행하고 출력과 점수를 기준 결과와 비교한다.
3. TensorRT 사용 시 엔진 호환성을 확인하고, 필요한 경우 대상 Jetson에서 재생성 후 비교한다.
4. 7절의 실행 폴더에서 복구용 venv로 GUI를 열어 모델 선택·촬영·추론·CSV/이미지 저장을 확인한다.
5. no_added_awb 조건의 일반 리셋과 USB 재연결 후 재측정을 각각 수행한다.
6. 네트워크를 끊고 재부팅한 뒤에도 같은 작업이 완료되는지 확인한다.

복구 완료 기록에는 사용한 백업 해시, OS/L4T·Python·GPU 라이브러리 버전, Git 커밋, 모델/엔진 해시, 설치 누락 및 보충 내역, 동일 입력 비교와 카메라 시험 결과를 남긴다. 현재 확인된 것은 **사용자의 백업 생성 완료 보고**이며 위 복구 시험 완료는 아니다.

## 4. GUI 포팅 범위 및 구조

| 영역 | 기존 Windows | Jetson 포팅 |
|---|---|---|
| 메인 GUI | 제공된 MOSA_visualAD_comb.py | MOSA_visualAD_comb_jetson.py |
| 화면·업무 흐름 | 모델 선택, 미리보기, 측정, 결과 표시·저장 | 기존 구조를 최대한 유지 |
| 영상 취득 | OpenCV DirectShow | OpenCV V4L2 |
| 장치 제어 | DNX64.dll | uvcdynctrl XU 명령 + v4l2-ctl 영상 제어 |
| 카메라 실행 주체 | 기존 GUI/캡처 코드 | 단일 카메라 worker가 읽기·설정·재연결 담당 |
| 모델 실행 | 기존 VisualAD ONNX wrapper | 공통 인터페이스의 ONNX CUDA / TensorRT backend |
| 전처리 | 기존 MOSA 및 get_transform | 원본 흐름 유지, 정규화 중복 적용 방지 |
| 후처리 | VisualAD 특징→맵·점수 | 공유 visualad_adapter |
| SubspaceAD | 제공본의 비활성 코드 | 주석 처리된 흐름 유지, 불필요한 시작 로드 비활성화 |
| 설정 | data.json + 개발 중 별도 설정 파일 | 메인 운용 설정은 data.json으로 통합 |
| 측정 로그 | 기존 CSV | 실행 backend 추가, 기존 행은 unknown으로 보존 |

핵심 흐름은 다음과 같다.

```text
모델 선택 → 추론 backend 로드 → 카메라 초기화 → 미리보기
  → 측정 클릭 → 설정/영상 조건 검사
      ├─ 통과 → 촬영 영상 전처리 → 추론 → 후처리 → 결과 표시·저장
      └─ 미달 → FAIL(WAIT) → 재초기화 → READY → 사용자가 다시 측정
```

MOSA 메인은 전체 프레임 평균화를 사용한다. 과거 test.py 및 독립 진단 코드의 ROI만 평균화하는 경로와 구분한다. 추론은 기존 GUI 흐름을 유지한 동기 실행이며, GUI 전체를 비동기 구조로 다시 작성한 것은 아니다.

## 5. ONNX / TensorRT 추론 연결 및 검증

### 5.1 모델 입출력

실제 ONNX 입력은 `input`, float32 `[1,3,336,336]`이다. 출력은 다음 7개이며 최종 anomaly score가 아닌 특징 텐서다.

| 출력 | 크기 |
|---|---|
| anomaly_features_enhanced | [1,1024] |
| normal_features_enhanced | [1,1024] |
| class_features | [1,1024] |
| patch_tokens_transformed_layer_6 / 12 / 18 / 24 | 각각 [1,579,1024] |

원본 wrapper와 동일하게 각 레이어의 앞 3개 특수 토큰을 제외하고 576개 토큰으로 24×24 맵을 만든다. 이상/정상 특징과의 코사인 유사도 차이를 구하고 336×336으로 보간해 네 레이어 맵을 합산한다.

점수는 **Gaussian 적용 전 합산 맵의 상위 0.5%(TopK 비율 0.005) 평균**이다. Gaussian sigma=4는 표시 맵·마스크에 적용한다. 점수는 0~1 확률로 제한되는 값이 아니며, 임의 정규화로 Windows 수치에 맞추지 않는다.

### 5.2 검증 결과와 해석

- ONNX Runtime 1.24.0에서 CUDA provider와 실제 입력 실행을 확인했다.
- 0 입력 `[1,3,336,336]` 추론이 종료 코드 0으로 완료되고 출력 NPZ가 생성됐다.
- ONNX와 TensorRT의 raw 특징 비교에서는 일부 원소가 당시 허용오차를 초과했다. 마지막 레이어에서 약 0.1149%가 초과했고 최대 절대오차는 약 7.82e-4였다. 따라서 raw 특징이 완전히 동일하다고 보고하지 않는다.
- Windows와 Jetson의 동일 BMP 비교에서 처음에는 히트맵은 유사하나 점수 차이가 있었다. 별도 어댑터의 TopK 비율을 원래 값 0.005로 맞춘 뒤 사용자가 결과 일치를 확인했다.
- 동일 BMP 검증은 제공된 비교 사례의 결과다. 전체 제품군·모든 임계값 부근 사례·모든 엔진 정밀도의 정확도를 포괄하는 인수 시험은 아니다.
- 사용자는 ONNX와 TensorRT 성능을 확인했고 ONNX도 충분히 빠르다고 평가했다. 표준화된 반복 latency/FPS 수치는 제공되지 않아 정량 가속 배수는 기재하지 않는다.

TensorRT 엔진은 대상 Jetson에서 `scripts/build_engine.sh`로 생성하고, ONNX와 동일 입력을 사용해 비교하도록 도구를 구성했다. 모델별 `model_A_path_engine`~`model_E_path_engine`을 기존 ONNX 모델 선택 항목과 대응시켰다. 실행 중 backend만 바꿔도 세션이 바뀌지는 않으며 GUI 재시작이 필요하다.

## 6. Dino-Lite 제어 이전과 색감 차이 해결 과정

### 6.1 기본 제어

사용자가 다음 LED 명령의 동작을 확인했다.

```bash
uvcdynctrl -d video0 -S 4:2 f2000000000000  # OFF
uvcdynctrl -d video0 -S 4:2 f2010000000000  # ON
```

AE ON/OFF, 고정 노출 명령, 밝기 설정 및 조건 미달 시 복구를 순차적으로 검증했다. Windows `ExposureValue=800`과 실제 노출시간의 변환식은 확인되지 않았으며, 이를 800ms라고 해석하지 않는다. Windows 800 프로필은 캡처된 명령 시퀀스를 재현한다. 실제 노출 readback은 현재 제공하지 않는다.

밝기 검사 변수 이름과 실제 채널도 확인했다. 원본의 `bright_b`는 RGB 변환 후 **R 채널 평균**이며, RG gap은 `int(R-G)`다. 원본 처리에 맞춰 하단 10줄 영역을 검사한다. Brightness 제어값과 영상 R 평균은 서로 다른 값이다.

### 6.2 Windows 리셋 캡처 분석

Windows 로그의 한 사례에서 리셋 전 RGB는 약 `182.0 / 163.2 / 142.0`, RG gap은 18이었다. 리셋 후 수동 재측정은 `171.1 / 166.0 / 136.9`, gap 5, anomaly score 약 5.22475였다.

해당 리셋 USB 캡처에서는 다음 순서가 관측됐다.

1. 기존 스트림 중지.
2. YUY2 640×480 30fps 스트림 시작·종료를 두 번 수행.
3. MJPEG 2592×1944 10fps 스트림 시작.
4. XU/PU SET_CUR 쓰기 78개 수행.

기존 full 재생과 명령 데이터는 같았지만 간격이 달랐다. 성공 캡처의 간격으로 갱신하고 스트림 전환을 V4L2로 근사하는 옵션을 추가했다. 다만 이 변경만으로는 Jetson RG gap 17이 개선되지 않았다는 사용자 결과를 받았다. 물리 LED의 세 번 점멸 원인도 스트림 전환과 관련됐을 가능성까지만 확인했다.

### 6.3 추가 AWB 쓰기 분리

Windows 캡처 이외에 Jetson 코드가 수행하던 제어 쓰기를 단계적으로 분리했다.

| 시험 모드 | 변경 | 사용자 결과 |
|---|---|---|
| baseline | 재생 전 AWB OFF + 재생 후 영상 제어 재쓰기 | 기존 비교 기준 |
| no_post_writes | 재생 후 중복 쓰기 생략 | 이후 추가 AWB 분리 시험으로 진행 |
| no_added_awb | 재생 전 AWB OFF도 생략 | 색감 문제가 잡혔으며 리셋 후 수동 재측정 경향이 일관적이라고 확인 |

이 결과는 추가 AWB 쓰기의 영향에 대한 근거다. Windows가 리셋 뒤 반드시 AWB ON으로 동작했다는 결론은 아니다. Windows 캡처에서는 AWB OFF 조회가 있었으며 내부 색 보정 상태가 어떻게 바뀌는지는 별도 확인 대상이다. 개선 후 RGB·점수 전체 수치 표는 아직 제공되지 않았다.

### 6.4 물리적 USB 재연결 복구

USB 재연결 뒤 AWB=1, white_balance_temperature=5800이지만 inactive 상태가 확인됐다. 이때 수동 WB 설정이 Permission denied로 실패했다. 다음 대응을 구현했다.

- 리셋 초반 기존 카메라 핸들 해제 후 최대 10초 동안 USB 장치 재등록 대기.
- VID/PID와 영상 노드를 확인해 변경된 `/dev/videoN`으로 캡처·제어 경로 갱신.
- no_added_awb에서 수동 WB 직전 AWB를 조회하고, ON일 때만 OFF 전환·readback 확인.
- AWB가 이미 OFF인 일반 리셋에서는 불필요한 OFF 쓰기를 생략.

후속 사용자 로그에서 장치 재탐색, 세 번의 스트림 시작, 조건부 AWB 해제, 영상 제어값 readback까지 확인했다. USB 재연결 후 RG gap·점수의 반복 수치 검증은 남아 있다.

## 7. 현재 운용 설정 및 실행

아래는 기존 data.json에 병합하는 카메라·런타임 항목이다. 완전한 data.json이 아니므로 모델·저장·검사 임계값 등 기존 키를 보존해야 한다.

```json
{
  "backend": "onnx",
  "provider": "CUDAExecutionProvider",
  "camera_profile": "windows_800_full",
  "windows_control_trial": "no_added_awb",
  "windows_stream_restart": true,
  "ExposureValue": 800,
  "Brightness": 16,
  "reset_flag_en": 1,
  "hide_nvmap_messages": true
}
```

영상 제어 기준값은 contrast16, hue0, saturation32, sharpness0, gamma5, AWB0, WB5800, power_line_frequency2다. no_added_awb는 이 값을 재생 후 중복해서 쓰지 않는다. 조건 검사 readback은 유지한다. 실제 BRIGHT_min/max와 RG_gab는 업무 기준값을 사용하고, 통과를 위해 임의 완화하지 않는다.

현재 사용자의 폴더 구성에서 실행 예시는 다음과 같다.

```bash
# 저장소 루트에서 venv 활성화 후 data.json이 있는 폴더로 이동
source .venv/bin/activate
cd MOSA_comb/MOSA_comb
python ../../scripts/run_mosa_logged.py
```

직접 `python MOSA_visualAD_comb_jetson.py`로도 실행할 수 있으나 NvMap 표시 필터는 별도 실행기에만 적용된다. 원본 출력 로그는 `artifacts/runtime-logs/`에 저장한다.

## 8. 오류·경고 및 운영상 처리

| 항목 | 확인 내용 | 현재 처리 |
|---|---|---|
| libcudss.so.0 import 실패 | 설치 라이브러리의 검색 경로 누락 | 경로 지정으로 해결 확인 |
| JSON load failed | 잘못된 설정 파일 수정, 마지막 요소 쉼표 사례 | 실행 폴더/data.json 확인 및 문법 수정 |
| 고정 노출 미선택 | 초기 fixed 모드 설정 누락/불일치 | 검증된 설정 사용, Windows 프로필과 구분 |
| WB Permission denied | USB 재연결 후 AWB ON, 수동 WB inactive | 수동 WB 직전 조건부 AWB 해제 구현 |
| NvMap 할당 오류 출력 | 최초 동작에서 관측, 이후 추론 완료 사례 있음 | 사용자 판단으로 현 단계 비차단 관찰 항목; 선택적 터미널 숨김, 원본 로그 보존 |
| JPEG premature end 경고 | 재연결/스트림 동작 중 관측 | 현 단계 작업을 막지 않는 관찰 항목; 경고 시각·횟수 기록 |
| usbmon 모듈 없음 | CONFIG_USB_MON is not set 확인 | 현재 커널에서 USB 캡처 불가, 커널 변경 보류 |

NvMap 메시지 표시 억제는 할당 오류를 해결한 것이 아니다. JPEG 경고도 원인이 해결됐거나 무해성이 입증된 것은 아니다. 현재 사용자 방침에 따라 개발·운용을 계속하면서 기록하고 있으며, 손상 프레임 자동 폐기 기능은 아직 구현되지 않았다. 실제 영상·판정에 영향이 관찰되면 재분석한다.

L4T 업데이트 이후 카메라를 사용하지 않는 ONNX 단독 시험에서도 NvMap 메시지가 발생했으나 종료 코드 0과 결과 파일 생성을 확인했다. 사용자는 비교 결과가 동일하다고 보고했다. 이 사실만으로 시스템 업데이트가 원인이라고 확정할 수는 없다.

## 9. 검증 현황

| 검증 항목 | 현황 | 근거·한계 |
|---|---|---|
| Jetson GUI 실행 | 사용자 확인 | 실제 모델과 사내 리소스를 사용한 메인 실행 |
| ONNX CUDA 추론 | 사용자 확인 | 실제 입력 실행 및 결과 저장 |
| TensorRT 추론 | 사용자 확인 | 실행·성능 확인, 정량 벤치마크 표는 미작성 |
| 동일 BMP의 Windows/Jetson 결과 | 사용자 일치 확인 | TopK0.005 정렬 후, 확인 사례 범위 |
| 카메라 LED·AE·밝기 제어 | 사용자 확인 | 단계별 실제 장비 시험 |
| 조건 미달 복구·추론 차단 | 사용자 확인 및 코드 구현 | before/after와 수동 재측정 흐름 |
| no_added_awb 리셋 후 경향 | 사용자 반복 일관성 확인 | 상세 반복 수치 표는 추가 수집 필요 |
| 물리 재연결 장치 탐색·WB 설정 | 사용자 로그 확인 | 색감·점수 인수 검증과는 별개 |
| 카메라/GUI 관련 로컬 테스트 | 40개 통과 기록 | 장치 재탐색/AWB 복구 변경 시 모의 테스트 |
| 로그 실행기 테스트 | 3개 통과 기록 | 네이티브 출력 수집·필터·종료 코드 확인 |
| 장시간 연속 운전·완전 오프라인 인수 | 추가 검증 필요 | 완료 보고 없음 |
| wheel-backup 생성 | 사용자 완료 보고 | 백업 직접 검사·신규 환경 복구 시험은 미수행 |
| 새 SD에서 전체 재설치 재현 | 추가 검증 필요 | 설치 이력·절차 정리와 별도 검증 |

로컬 테스트 수는 각 변경 시 실행한 대상별 결과이며, Jetson 전체 시스템 인수 시험의 총 통과 건수로 합산하지 않는다.

## 10. 후속 계획

1. 동일 조명·시료·배율·거리에서 Windows/Jetson 각각 반복 RGB, RG gap, score 및 판정을 표로 기록한다.
2. 일반 리셋, GUI 재실행, USB 전원 재연결을 구분해 동일한 영상 조건이 복구되는지 확인한다.
3. 실제 업무 데이터로 임계값 주변 OK/NG 사례를 비교한다.
4. 생성한 wheel-backup을 별도 매체에 보관하고 3.5절에 따라 누락 의존성을 보충한다. SD 이미지·시스템 패키지·모델/엔진 해시·설정·Git 커밋과 함께 새 SD 복구를 검증한다.
5. 네트워크 차단 및 재부팅 후 모델 로드·촬영·추론·저장을 확인한다.
6. 필요할 때 정량 성능과 장시간 운전을 측정한다. TensorRT 저정밀도 최적화는 현재 후순위다.

보고서용 반복 측정 표 권장 열: 실행 환경, Git 커밋, 모델/엔진 해시, backend, 리셋 유형, 시험 모드, 조명/시료 조건, 측정 번호, R/G/B, RG gap, score, threshold, OK/NG, 소요시간, 경고 발생 여부.

## 11. 주요 산출물

| 파일 | 역할 |
|---|---|
| MOSA_visualAD_comb_jetson.py | 기존 MOSA 구조 기반 Jetson 메인 GUI |
| scripts/dinolite_camera.py | 단일 worker 카메라 취득·제어·재탐색·복구 |
| scripts/camera_conditions.py | 설정 검증 및 영상 조건 검사 |
| scripts/mosa_jetson_bridge.py | 기존 GUI와 카메라/추론 연결 |
| scripts/inference_backend.py | ONNX / TensorRT 실행 추상화 |
| scripts/visualad_adapter.py | 공통 특징 후처리·점수 계산 |
| scripts/windows_800_capture.json | 분석된 Windows 제어 쓰기 목록 |
| scripts/run_mosa_logged.py | 네이티브 출력 기록·NvMap 표시 필터 |
| scripts/build_engine.sh / onnx_probe.py / trt_probe.py | 엔진 변환 및 단독 실행 점검 |
| scripts/compare_backends.py / compare_visualad.py | 특징 및 후처리 결과 비교 |
| requirements.txt / requirements-inference.txt | Python 의존성 목록 |
| JETSON_MOSA_WORKFLOW.md | 신규 설치부터 실행까지 상세 절차 |
| WINDOWS_800_CHECK.md | Windows 리셋 재현·시험·JPEG 진단 절차 |

사내 원본 영상·모델·실제 설정·Windows PCAP/로그는 공개 저장소 산출물과 분리해 보관한다. 본 문서는 해당 자료에서 확인한 결과를 요약하며 원본을 포함하지 않는다.

## 12. 보고서 본문용 요약 문안

> Jetson Orin Nano Super 기반으로 기존 Windows MOSA 검사 GUI의 장치 내 실행 환경을 구축하였다. JetPack 6.2.1 SD 이미지 기반의 Python·GPU 실행환경을 구성하고, DNX64 DLL 및 DirectShow에 의존하던 카메라 제어를 Linux V4L2와 UVC 명령으로 이전하였다. 기존 GUI 구조와 전후처리를 유지하면서 ONNX Runtime CUDA 및 TensorRT 실행 경로를 연결하였다. 동일 BMP 비교를 통해 TopK 설정 정렬 후 Windows와 Jetson의 추론 결과 일치를 확인했으며, 촬영 영상의 차이는 Windows USB 리셋 분석과 추가 AWB 쓰기 제거를 통해 개선하였다. 현재는 일반 리셋 후 반복 측정의 일관성을 확인한 상태이며, 물리적 USB 재연결 후 영상 안정성, 신규 장비 재설치 재현성 및 오프라인 장시간 운전을 후속 검증 대상으로 관리한다.
