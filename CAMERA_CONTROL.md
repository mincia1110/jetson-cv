# Jetson 카메라 고정 설정 및 추론 전 검사

기존 test.py의 의도에 맞춰 **초기 밝기·수동 노출 적용 → 추론 직전 고정값 재적용 및 상태 검사 → 조건을 만족한 프레임만 전달**한다. AE로 조명에 다시 적응시키는 복구는 사용하지 않는다. 추론 모델과 기존 전처리는 변경하지 않았다.

## 설치·실행

인터넷 연결 가능한 준비 단계에서 설치한다.

```bash
sudo apt update
sudo apt install python3-venv python3-pip python3-tk python3-pil.imagetk python3-opencv python3-numpy uvcdynctrl v4l-utils
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p local
cp camera_check.example.json local/camera_check.json
python scripts/camera_control_gui.py --device /dev/video0 --config local/camera_check.json
```

이미 만든 가상환경은 활성화만 한다. `requirements.txt`는 카메라 GUI용이며 OpenCV·Tkinter·USB 도구는 apt로 설치한다. 기존 카메라 프로그램을 종료한 뒤 실행한다.

## 초기값 입력

화면에 **초기 밝기(V4L2)**와 **고정 노출 시간**을 입력하고 **초기 고정값 적용**을 누른다. 설정 파일에 Brightness와 ExposureTime이 있으면 시작할 때 자동 적용한다. 예제의 null 값은 사용자가 아직 초기값을 지정하지 않았다는 뜻이다. 미입력 상태에서는 미리보기만 가능하고 측정은 허용하지 않는다.

- `Brightness`: 실제 장치의 Linux brightness 범위 안의 정수. `v4l2-ctl -d /dev/video0 --list-ctrls-menus`로 범위를 확인한다.
- `ExposureTime`: 드롭다운의 수동 노출 시간. 예: `1/60s`. Windows DLL의 ExposureValue 숫자와 자동 변환하지 않는다.
- `BRIGHT_min`, `BRIGHT_max`, `RG_gab`: 기존 test.py의 실제 영상 판정 기준을 넣는다. 예제의 넓은 범위는 자리표시자이며 실제 합격 기준이 아니다.
- `capture_no`: 검사·추론용 ROI 평균 프레임 수.
- `settle_frames`: 설정 적용 뒤 버릴 프레임 수. 기본 5. 설정이 실제 프레임에 반영되는 시간은 실물에서 확인한다.
- `reset_flag_en`: true이면 영상 조건 불일치 시 한 번 재연결·고정값 재적용 후 재판정한다. false여도 매 측정 전 고정값 적용과 검사 자체는 실행한다.
- `exposure_reset_mode`: `fixed`. 이전 `ae_relearn` 설정은 제거한다.

밝기·노출·프레임 수는 화면 입력값을 사용한다. 다른 판정 기준은 버튼을 누를 때마다 JSON에서 다시 읽는다. 화면에서 바꾼 값은 파일에 자동 저장하지 않으므로 다음 시작에도 적용하려면 local JSON을 수정한다. `local/`과 PROGRESS.md는 Git에서 제외된다.

## 측정 흐름

**상태 확인 후 촬영** 버튼과 API `prepare_inference()`는 같은 경로를 사용한다.

1. 현재 V4L2 brightness 조회.
2. LED OFF → AE OFF → 지정한 수동 노출 명령 전송 → 초기 brightness 적용.
3. brightness를 다시 조회해 목표값과 비교. 실패하면 측정 중단.
4. 설정 반영용 프레임을 버리고 새 프레임 N장 취득. test.py처럼 `[900:1100, 0:2590]` ROI만 평균화하고 바깥은 첫 프레임 유지.
5. RGB 변환·7×7 Gaussian blur 후 `[1934:1944, 0:2590]` 영역 검사. 기존 bright_b는 실제로 `int(R 평균)`이고 RG_diff는 `int(R 평균 - G 평균)`이다. 경계값은 합격이다.
6. 영상 조건이 벗어나고 reset_flag_en이 true이면 재연결 후 동일한 고정값으로 한 번만 재시도.
7. 조건을 만족한 **검사 대상 프레임 자체**를 반환. 실패하면 frame=None으로 추론·이미지 저장을 차단.

화면·터미널에는 적용 전 brightness, 적용 후 측정값, 리셋 여부, 실패 이유, 추론 허용 여부를 표시한다. `조건 검사·자동 복구`는 저장 없이 동일 검사 과정을 실행한다. LED·AE 수동 버튼은 실험용이고 다음 측정에서 고정 설정으로 되돌린다.

## 노출 제어의 확인 범위

AE ON/OFF·노출 유지·LED ON/OFF는 AM7115MZT(a168:0960)에서 사용자 확인 완료다. 수동 노출 시간 명령은 [공개 5MP 명령표](https://github.com/miked63017/dinolite_uvc_led_control/blob/master/Notes.txt)를 사용했다. **AM7115MZT에서 각 수동 노출값의 실제 적용은 아직 실물 확인이 필요하다.**

노출값을 읽어오는 검증된 프로토콜은 없어서 매 측정마다 지정 명령을 다시 보낸다. brightness는 실제 조회·비교하지만 노출은 명령 전송만 확인할 수 있다. `exposure_verified: false`는 노출 readback이 없다는 의미다. `IMAGE_AND_BRIGHTNESS_OK`를 특정 셔터 시간 검증 완료로 해석하지 않는다. DLL ExposureValue 숫자를 알고 있더라도 해당 숫자와 노출 시간의 대응은 별도로 확인해야 한다.

## test.py 추론에 연결

새 모듈은 아직 기존 test.py 전체를 자동 교체한 파일이 아니다. 기존 DLL 참조, 중복 camera.read() 루프와 reset_camera 대신 이 모듈을 연결한다. GUI 스레드는 Future를 기다리지 말고 root.after로 완료를 확인한다.

```python
camera = DinoLiteCamera('/dev/video0', initial_config=config)
# camera.ready 완료 뒤 측정 요청
pending = camera.prepare_inference(config)
# 이후 root.after 콜백에서 pending.done() 확인
result = pending.result()  # 완료된 경우에만 호출
if result['frame'] is not None:
    frame_to_save = result['frame']  # uint8 BGR; 기존 전처리 입력
    # 기존 RGB 변환/전처리 -> user_th_inference(...) 실행
else:
    # FAIL 표시, 추론 실행하지 않음
    pass
```

prepare_inference가 반환한 프레임을 사용한다. 이후 latest()나 별도 camera.read()로 다시 취득하면 검사한 프레임과 달라진다. 무거운 추론은 별도 작업 스레드에서 수행하고 UI만 메인 스레드에서 갱신한다. 종료는 camera.close() 요청 후 is_alive()를 폴링한다.

## Jetson 실물 확인

1. 고정 조명에서 밝기와 노출을 입력하고 적용. 노출 시간을 두 가지로 바꿔 영상 변화를 확인한다.
2. JSON에 실제 밝기·R−G 범위를 입력하고 정상 조건에서 촬영·저장이 되는지 확인한다.
3. 실험용 AE ON이나 다른 brightness를 적용한 뒤 측정해 초기 고정값으로 돌아오는지 확인한다.
4. 빛을 바꿔 기준 밖으로 만들고 측정. 한 번 재연결해도 기준 밖이면 FAIL·추론 차단이어야 한다. AE로 조명에 적응해 통과시키지 않는다.
5. 재연결·종료 중 충돌이 없는지 확인한다. 드라이버 read()가 멈추면 종료 지연이 가능하다.

요청 영상 모드는 MJPG 2592×1944, 기본 10fps다. 장치가 다른 크기를 반환하면 ROI 오류를 막기 위해 중단한다. 긴 노출 시간에서는 프레임 수신과 종료가 느려질 수 있다. 로컬 테스트는 모의 장치 기반이며 실제 USB·GUI·노출 적용 검증을 대체하지 않는다.
