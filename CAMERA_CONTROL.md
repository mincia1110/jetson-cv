# Jetson 카메라 제어 개선판

`scripts/camera_control_gui.py`는 test.py의 카메라 부분을 분리한 Tkinter 검증 화면이다. 모델은 로드하지 않는다. 기존 test.py의 추론·전처리는 수정하지 않았다.

## 실행

Jetson에서 기존 캡처 프로그램을 종료하고 실행한다. 설치는 인터넷 연결 가능한 준비 단계에서 한다.

```bash
sudo apt update
sudo apt install python3-venv python3-pip python3-tk python3-pil.imagetk python3-opencv python3-numpy uvcdynctrl v4l-utils
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import cv2, numpy, tkinter; from PIL import Image, ImageTk; print('Camera imports OK')"
python3 scripts/camera_control_gui.py --device /dev/video0
```

이미 준비한 system-site-packages 가상환경에서도 실행할 수 있다. 기본 요청은 MJPG 2592×1944, 10fps다. 전달받은 장치 descriptor의 해당 모드에 맞췄으며 화면에 실제 보고된 FPS를 표시한다. 실제 프레임 크기가 다르면 ROI 오동작을 막기 위해 오류 처리한다.

- 시작 및 재연결 시 LED OFF를 보낸다.
- AE ON으로 광량에 적응시킨 뒤 AE OFF로 현재 노출을 고정한다. 이 동작은 사용자가 실물 검증했다.
- 시작 시 AE 상태는 강제 변경하지 않는다. 버튼으로 선택한 AE 모드는 재연결 시 다시 적용한다. **재연결 후 이전 노출 수치까지 복원된다는 보장은 없다.** 필요하면 다시 AE ON → 조명 안정화 → AE OFF를 수행한다.
- 촬영은 새 프레임 N장을 순차 취득한다. test.py와 동일하게 `[900:1100, 0:2590]` ROI만 평균화하고 나머지는 첫 프레임을 유지한다. uint8 BGR BMP로 저장한다.
- 재연결은 스트림 종료·재개방·설정 재적용이다. USB reset이나 공장 초기화가 아니다.
- 밝기는 검증한 Linux 값이 있을 때만 `--brightness 값`을 지정한다. DLL 설정 번호·노출 숫자를 그대로 옮기지 않는다.

LED와 AE 명령은 `dinolite_camera.py`에 모여 있다. FLC·AXI·EFLC·렌즈 위치·MicroTouch 등 기존 DLL 초기화 항목은 자동 적용하지 않는다. 자체 LED를 쓰지 않는 현재 범위에서는 LED OFF를 사용한다.

`requirements.txt`는 카메라 화면용 Python 의존성만 포함한다. OpenCV와 Tkinter, USB 제어 도구는 위 apt 명령으로 설치한다. 이미 `.venv`가 system-site-packages 방식으로 준비되어 있다면 생성 명령은 생략한다. 이 파일은 전체 추론 환경의 의존성이나 고정 버전 잠금 파일은 아니다.

## 기존 test.py에 연결할 경계

기존 `camera_capture_loop`, `camera_init`, `reset_camera`의 직접 VideoCapture/DLL 사용을 카메라 모듈로 대체한다. DLL을 참조하는 load_json과 CSV 로그도 분리해야 한다. GUI 전체를 그대로 두고 import만 바꾸는 교체품은 아니다.

```python
from dinolite_camera import DinoLiteCamera

camera = DinoLiteCamera('/dev/video0')
# GUI의 root.after 콜백에서 future.done()을 확인한다.
# GUI 스레드에서 아직 끝나지 않은 future.result()를 기다리지 않는다.
initialization = camera.ready
frame, error = camera.latest()                 # 미리보기 복사본
request = camera.capture(count=5)              # Future -> uint8 BGR 전체 프레임
reset_request = camera.reconnect()             # Future -> 실제 해상도/FPS
camera.close()                                # 비동기 종료 요청
```

위 호출은 API 예시이며 한꺼번에 실행하는 초기화 코드가 아니다. 실제 버튼·상태 처리는 검증 GUI를 참고한다. 촬영 future가 완료된 뒤 결과를 기존 전처리에 전달하고, 추론은 별도 작업 스레드에서 실행한다. 카메라의 `read()`와 `release()`는 외부에서 호출하지 않는다.

## 실물 확인 순서

1. 연결 완료, 2592×1944 영상, 시작 시 LED OFF 확인.
2. LED ON/OFF, AE ON → 광량 변화 → AE OFF → 광량 원복 후 노출 유지 확인.
3. 평균 프레임 수 1과 5로 촬영하고 BMP 크기·색상·ROI 확인.
4. 미리보기 중 재연결 후 영상과 LED OFF 확인. AE 고정 노출은 필요 시 재설정.
5. 케이블 분리 시 오류 표시와 오래된 프레임 제거 확인. 재연결 후 장치 경로를 확인하고 재연결 버튼 사용.
6. 촬영·재연결 중 창을 닫아 장치가 반환되는지 확인.

현재 호스트에서는 모의 카메라 테스트만 가능하다. V4L2 드라이버의 read() 자체가 멈추면 종료도 지연될 수 있다. 다른 스레드에서 강제 release하지 않으며, 이 경우 프로세스 종료 후 장치 상태를 점검한다.
