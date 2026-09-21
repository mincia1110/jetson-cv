# Windows ExposureValue 800 호환 모드 시험

목적: Windows 캡처에서 관측한 노출 800 시퀀스를 재현한다. 센서 노출시간 800ms를 뜻하지 않는다. 완전한 DLL/PCAP 재생은 아니다.

## 설정

기존 data.json을 별도 이름으로 백업한 뒤 다음 항목을 병합한다. 기존 모델/저장/임계값 키는 보존한다.

```json
"camera_profile": "windows_800",
"ExposureValue": 800,
"Brightness": 16,
"reset_flag_en": 1,
"video_controls": {
  "contrast": 16,
  "hue": 0,
  "saturation": 32,
  "sharpness": 0,
  "gamma": 5,
  "white_balance_automatic": 0,
  "white_balance_temperature": 5800,
  "power_line_frequency": 2
}
```

windows_800에서는 ExposureTime 항목이 남아 있어도 명령에 사용하지 않는다. 내부 로그는 DNX64:800으로 표기한다. 그 외 ExposureValue는 거부한다. video_controls에 기존 조정값이 있으면 해당 값이 우선하므로 위 비교값으로 명시적으로 교체한다. 영상 검사 범위와 임계값을 통과시키려고 임의 변경하지 않는다.

## 실행 흐름

- 처음 적용 및 재연결: LED OFF → AE ON → 0.5초 → AE OFF → 노출 다섯 쓰기(각 40ms 간격) → 0.5초 → 캡처의 AE-target 관련 네 쓰기 → FLC ON/레벨1/모든 사분면 OFF → 밝기 → 대비/색조/채도/선명도/감마 → AWB OFF/5800 → 60Hz → readback 및 안정화 프레임.
- MOSA 일반 측정: 설정이 그대로라면 노출/WB/LED 쓰기를 생략하고 새 프레임과 제어 readback을 검사한다. 설정값을 바꿨을 때만 재적용한다. 리셋 직후 상태를 다음 측정의 쓰기로 바꾸지 않기 위한 흐름이다. 독립 카메라 진단 GUI의 기존 매회 재적용 동작은 유지된다.
- MOSA 실패: reset_flag_en=1이면 FAIL(WAIT) → 스트림 닫기 → 1초 대기 → 재개방·전체 초기화 → 진단용 after 캡처 → READY(회색) → 수동 재측정. 실패한 클릭에서는 after가 통과해도 추론/결과 저장을 하지 않는다. after가 기준 미달이어도 초기화와 프레임 취득이 완료됐다면 READY이며, 이것은 영상 조건 통과가 아니다. 다음 클릭에서 다시 검사하고 미달이면 다시 추론을 차단한다. 재개방·명령·프레임 취득 자체가 실패하면 파란 FAIL로 남는다.
- 대기 0.5초는 제공된 Python 코드 기준이다. 실제 PCAP 간격과 완전히 일치하지 않으며 40ms도 캡처의 35~38ms 간격을 근사한 값이다.
- 미확인 AXI/EFLC/렌즈 쓰기·내부 조회·장치 열기/포맷 협상은 복제하지 않는다. LED ON 패킷도 그대로 재생하지 않고 조명 OFF 목적을 유지한다. 정확한 영상 동등성은 실물 검증 대상이다.

## 사용자가 확인할 순서

1. Windows를 ExposureValue800/Brightness16 및 제공된 나머지 기본값으로 실행한다. 외부 조명·시료·배율·거리·ROI를 고정하고 RGB와 anomaly score를 5회 기록한다.
2. Jetson에서 ONNX backend로 앱을 재시작한다. 프로필 전환 비교는 재시작한다. 같은 물리 조건에서 5회 측정한다.
3. 터미널 Camera check 보고서의 camera_profile=windows_800, ExposureTime_requested=DNX64:800, video_controls readback 및 allow_inference를 확인한다. 노출 readback은 여전히 unavailable이다.
4. `v4l2-ctl -d /dev/video0 --list-ctrls-menus`에서 brightness16, contrast16, hue0, saturation32, sharpness0, gamma5, AWB0, WB5800을 확인한다. 자체 LED가 꺼져 있는지도 확인한다.
5. 연속 측정에서 RGB가 유지되는지 확인한다. 이상 광량을 넣어 복구를 발생시킨 뒤 광량을 원복하고, 정상 조건의 다음 측정이 원래 RGB로 돌아오는지 확인한다. 비정상 조명을 유지하면 READY로 복귀하더라도 영상 조건 미달과 추론 차단이 계속돼야 한다.
6. 결과 공유: Windows/Jetson 각각 RGB 평균과 범위, score 평균과 범위, before/after 상태. BMP 반출은 필요 없다.

## 리셋 후 수동 재측정 비교

1. 메인 파일과 `scripts/`를 함께 갱신하고 앱을 재시작한다. 사용하는 data.json의 reset_flag_en=1을 확인한다. 한 프로필씩 시험한다.
2. 최초 측정의 `[Camera check]` 보고서에서 `settings_action=preserved`를 확인한다. 밝기/영상 속성 등 요청 설정을 변경한 클릭에서는 `reapplied`일 수 있다.
3. 실제 영상 조건 미달로 리셋이 발생하면 `reset_performed=true`, `reset_succeeded=true`, `manual_retry_required=true`, `allow_inference=false`를 확인한다. `before`/`after`의 RGB와 RG_diff를 기록한다. `image_conditions_ok`가 after 영상의 진단 결과다.
4. 회색 READY에서 외부 조명·시료를 Windows 비교 조건으로 고정하고 직접 측정을 누른다. 이 클릭의 `settings_action=preserved`와 `before` RGB/RG_diff를 기록한다. 통과하면 이 클릭에서만 추론한다.
5. 설정을 보존한 재측정에서도 Windows와 차이가 남으면 해당 Windows 실행의 GUI 시작→조건 미달→reset→READY→수동 재측정까지 USB 캡처한다. 제어 명령/읽기 응답/간격을 대조한다. 이번 변경만으로 색감 동등성을 보장하지 않는다.

노출은 여전히 readback 불가다. preserved는 소프트웨어가 불필요한 쓰기를 생략했다는 뜻이며, 카메라 내부 상태가 실제로 고정됐다는 보증은 아니다. 비교 중 외부 도구로 제어값을 변경하지 않는다.

기존 모드 복귀: windows_stream_restart를 false로, camera_profile을 fixed로 바꾸고 기존 ExposureTime과 video_controls/Brightness를 백업에서 복원한 뒤 재시작한다. 프로필 이름만 바꾸면 나머지 장치 설정까지 자동 원복되지는 않는다.

노출 시퀀스: 0502000c357810 → 0525000d357810 → 05000000357810 → 05320001357810 → 05000002357810. GUI 실행 중 별도 uvcdynctrl 수동 명령을 병행하지 않는다. 설정 보존 모드에서는 외부의 노출 변경을 readback으로 감지하거나 매 측정에 덮어쓰지 못한다.

## 미확인 쓰기 포함 실험: windows_800_full

camera_profile을 `windows_800_full`로 바꾸고 앱을 재시작한다. ExposureValue800, Brightness16, 위 Windows 기본 video_controls를 사용해야 한다. 2026-09-18 성공 리셋 캡처의 장치2:24에서 관측한 성공한 SET_CUR 쓰기 78개(XU unit4 및 PU unit3)를 원래 순서·반복대로 재생한다. 데이터는 scripts/windows_800_capture.json에 있다. 의미 미확인 쓰기 및 조회 준비용 e0 쓰기도 포함한다. LED ON 명령도 포함되므로 초기화 동안 LED가 일시적으로 켜질 수 있다.

USB 요청/응답을 통째로 복제하는 모드는 아니다. 표준 USB 요청·GET 응답·영상은 제외한다. 아래 windows_stream_restart 옵션으로 영상 포맷 전환을 V4L2에서 근사할 수 있다. 따라서 e0 쓰기 뒤 원래 DLL이 수행한 GET까지 재현한 것은 아니다. 패킷 간 타임스탬프 차이를 sleep으로 사용하므로 실제 명령 실행시간이 더해져 Windows와 정확한 타이밍은 다르다. AWB는 캡처 조회 결과인 OFF를 재생 전에 명시적으로 설정한다. 재생 후 기존 고정 영상 속성을 재적용하고 readback·안정화한다.

MOSA 일반 측정에서는 설정이 그대로면 제어 쓰기를 생략한다. 설정 변경 시에는 AE OFF 및 노출 다섯 쓰기와 영상 속성을 적용한다. 재연결 복구 때는 전체 78개를 재생한다. 실패 시 정상 프레임으로 간주하지 않고 위 수동 재측정 흐름을 따른다. 프로필 전환은 반드시 재시작한다.

비교: windows_800에서 5회 RGB/score → windows_800_full로 재시작 후 동일 조건 5회 → 복구 후 5회. 자체 조명이 최종적으로 꺼졌는지 확인한다. 전체 옵션에서만 결과가 가까워지면 미확인 설정의 영향으로 범위를 좁힐 수 있지만 개별 명령 의미가 확정된 것은 아니다.


## 2026-09-18 성공 리셋의 스트림 전환 시험

Windows 로그는 리셋 직전 RGB=(181.9985,163.2368,142.0183), RG_diff=18에서,
리셋 후 수동 재측정 RGB=(171.1054,165.9932,136.8743), RG_diff=5로 변했다.
후자의 anomaly score는 5.224750이다. 같은 BMP에서 추론 동등성은 사용자가 이미 확인했다.

PCAP에서 한 번의 reset 중 다음 순서가 확인됐다. 시간은 캡처 시작 상대 초다.

| 프레임 | 시간 | 동작 |
|---|---:|---|
| 8140 | 39.1996 | 기존 스트림 STOP |
| 8156/8158 | 40.6430/40.6434 | YUY2 640×480 30fps COMMIT/START |
| 8196 | 40.9206 | STOP |
| 8212/8214 | 41.6738/41.6742 | YUY2 640×480 30fps COMMIT/START |
| 8252 | 41.9538 | STOP |
| 8268/8270 | 42.7036/42.7040 | MJPEG 2592×1944 10fps COMMIT/START |
| 8372–9418 | 43.3857–47.4759 | XU/PU SET_CUR 78개 |

카메라 Configuration Descriptor의 format/frame index와 COMMIT interval을 함께 해석했다.
최종 MJPEG는 format2/frame12, interval=1000000×100ns=0.1초다.
중간 YUY2는 format1/frame1, interval=333333×100ns다.
78개 쓰기의 데이터/순서는 기존 캡처와 동일하다. 이번 자료에서는 AE ON→OFF 간격이
약 0.611초, 마지막 노출 쓰기→AE target 간격이 약 0.537초다.
full 목록의 간격을 이번 자료로 갱신했다. 실행 도구의 오버헤드는 별도로 더해진다.

LED master ON(f201)은 이 구간에 한 번만 있다. 세 번의 점멸은 세 번의 스트림 시작과
연관됐을 가능성이 있지만, USB 캡처만으로 물리 LED 점멸의 원인을 확정할 수 없다.
GUI 시작 로그는 있지만 PCAP에는 초기 GUI 시작 전체가 들어 있지 않다.

기존 data.json에 다음 키를 병합하고 앱을 완전히 종료 후 다시 실행한다.

```json
"camera_profile": "windows_800_full",
"windows_stream_restart": true,
"ExposureValue": 800,
"Brightness": 16,
"reset_flag_en": 1
```

video_controls는 위의 Windows 기본값을 유지한다. 이 옵션은 windows_800에도 적용 가능하다.
옵션 생략/false면 중간 스트림 없이 기존 방식으로 연다. true면 시동 및 재연결마다
YUYV 640×480 30fps를 9프레임씩 두 차례 읽고 닫은 뒤 최종 MJPG를 연다.
중간 종료 뒤 0.72초, 최종 스트림 시작 뒤 0.68초 대기한다.
이는 OpenCV/V4L2를 통한 근사이며 Windows 드라이버의 내부 협상 및 종료 지연까지
일치시키는 것은 아니다. USB SET_INTERFACE를 직접 보내지는 않는다.
중간 해상도/포맷/fps 확인에 실패하면 조용히 대체하지 않고 실패를 보고한다.
중간 프레임은 미리보기·측정에 전달하지 않는다. 최종 모드는 2592×1944 10fps여야 한다.

1. 터미널에 `[camera reset] stream 1/3`, `2/3`, `3/3`이 나오는지 확인한다.
2. 동일한 정상 외부 조명·시료 상태에서 최초 측정 RGB/RG_diff를 기록한다.
3. RG_diff가 17–19인 상태에서 기존 검사 기준으로 리셋을 유발한다.
   Windows의 성공 사례처럼 **정상 조명 상태에서 RG 기준 미달로 리셋**하는 비교가 우선이다.
   리셋 중 손전등을 비추면 AE 수렴 조건까지 달라진다.
4. FAIL(WAIT)→회색 READY 후 직접 재측정한다. before/after와 다음 클릭의
   settings_action=preserved, RGB/RG_diff, score를 기록한다. 5회 반복한다.
5. 같은 설정에서 windows_stream_restart=false로 재시작하여 비교한다.
   full 명령 간격은 양쪽 동일하므로 중간 스트림의 영향을 분리할 수 있다.

RG_diff≈5 복구는 아직 Jetson에서 확인되지 않았다. 차이가 남으면 다음 후보는
Windows e0 조회의 GET 응답 왕복, 드라이버의 제어값 캐시, Linux에서 추가되는 AWB OFF/
power_line_frequency 및 영상 속성 재쓰기다. 해당 항목은 이번 패치에서 동시에 바꾸지 않았다.
원본 ZIP/PCAP/Windows 로그와 전체 분석은 ignored local/에 보관하며 공개 저장소에 넣지 않는다.

## 추가 제어 쓰기 분리 시험 (2026-09-21)

Jetson에서 스트림 전환 패치 이후에도 수동 재측정 RG_diff=17이라는 결과를 받았다.
또한 실행 커널 5.15.148-tegra의 CONFIG_USB_MON이 비활성화돼 usbmon 캡처는 현재 불가하다.
다음 비교에서는 스트림 옵션·노출·조명·시료·검사 임계값을 고정하고 추가 쓰기만 변경한다.
GET 왕복 재현과 커널 변경은 아직 적용하지 않았다.

`camera_profile: windows_800_full` 전용 `windows_control_trial` 옵션:

| 값 | 재생 전 AWB OFF | 재생 후 밝기/영상 제어 재쓰기 |
|---|---|---|
| baseline (생략 시 기본값) | 유지 | 유지 |
| no_post_writes (1단계) | 유지 | 생략 |
| no_added_awb (2단계) | 생략 | 생략 |

78개 캡처 쓰기, 스트림 옵션, LED 명령, 노출 명령/대기 시간은 세 단계에서 같다.
즉 no_added_awb도 순수 USB 재생을 뜻하지 않는다. Linux 드라이버의 내부 동작은 남는다.
재생 뒤에는 제어값을 읽고 검증한다. WB/전원 주파수 등을 목표와 맞추기 위한 추가 쓰기는 없다.
읽기 결과가 다르면 해당 값과 목표를 표시하고 실패 처리한다. 이를 색감 시험 성공으로
해석하거나 임계값을 완화하지 않는다. 2단계에서 USB 재연결 후 AWB가 ON이면, 수동 WB 쓰기 직전에만 OFF로 전환하고
readback=0을 확인한다. 이미 OFF이면 AWB 쓰기를 생략한다. OFF 전환 실패는 숨기지 않는다.

### 1단계: 재생 후 중복 쓰기 제거

기존 data.json에서 Windows 기본 video_controls는 유지하고 다음을 병합한다.

```json
"camera_profile": "windows_800_full",
"windows_control_trial": "no_post_writes",
"ExposureValue": 800,
"Brightness": 16,
"reset_flag_en": 1
```

windows_stream_restart는 직전 시험값을 그대로 둔다. GUI를 완전히 종료 후 재시작한다.
터미널 `windows_control_trial=no_post_writes`와 `post replay readback only`를 확인한다.
정상 외부 조명에서 최초 측정→기준 미달→FAIL(WAIT)→리셋→회색 READY→수동 재측정을
수행한다. 정상 측정 5회의 RGB/RG_diff/score와 before/after 보고서를 남긴다.
재측정 settings_action은 preserved여야 하며 보고서에 선택한 windows_control_trial이 나온다.
동일한 설정을 사용하는 진단 도구에서 재적용을 요청해도 실험 모드는 노출/WB를 다시 쓰지 않는다.
리셋/재연결 때는 선택한 단계로 전체 캡처 시퀀스를 다시 실행한다.

### 2단계: 재생 전 AWB OFF도 제거

1단계에서 RG_diff=17 부근이 유지되면 `windows_control_trial`만 `no_added_awb`로 바꾸고
GUI를 재시작해 동일한 과정을 수행한다. 1단계에서 개선되면 먼저 그 결과를 기록하고 반복
재현성을 확인한다. 실행 중 단계 전환은 비교에 사용하지 않는다.

원복은 `windows_control_trial: baseline`으로 변경 후 재시작한다.
설정 변경 중 카메라 내부 상태가 이월될 수 있으므로, 비교별 시작 조건(USB 재연결 여부,
Windows에서 사용 직후인지)을 함께 기록하고 동일하게 유지한다.
현재 실물 RG gap 개선은 미검증이다. 로그/패킷의 차이를 하나씩 분리하는 진단 옵션이다.


### USB 물리적 재연결 복구

사용자 검증: no_added_awb에서 일반 리셋 후 수동 재측정 경향은 일관적이다.
하지만 USB 재연결 후 AWB=1, white_balance_temperature=5800 flags=inactive 상태에서
수동 WB 쓰기가 Permission denied로 실패했다. 5800이라는 표시만으로 수동 WB가
활성화됐다고 판단할 수 없다.

no_added_awb는 캡처 마지막 수동 WB 쓰기 직전에 AWB 상태를 읽는다.
- AWB=0: AWB에 쓰지 않는다. 기존에 성공한 일반 리셋 경로를 유지한다.
- AWB=1: 그 시점에만 OFF를 한 번 쓰고 0 readback을 확인한 뒤 WB5800을 쓴다.
- 읽기 실패 또는 OFF 전환 실패: 재초기화를 실패 처리하고 수동 WB를 강행하지 않는다.

이 처리는 USB 탈착을 직접 탐지하는 것이 아니라 AWB 상태에 따른 복구다.
재생 전 AWB OFF나 재생 후 중복 쓰기는 다시 추가하지 않았다.
커널/장치 권한 변경은 필요하지 않다. USB 재연결 후 색감 동등성은 아직 미검증이다.

설정은 no_added_awb를 유지한다. GUI 실행 중 USB 재연결→측정→재초기화→READY→수동
재측정으로 확인한다. 터미널에 `AWB=1: disabling immediately before manual WB`가
나와야 하며, 이후 일반 리셋에서 AWB=0이면 이 메시지는 나오지 않아야 한다.
USB 재연결 복구 후의 RGB/RG_diff도 5회 기록한다. 다시 gap17이 되면 일반 리셋 성공과
USB 전원 재연결 복구를 별도 결과로 보고한다. 검사 기준은 완화하지 않는다.

리셋의 스트림 재개방 전에 기존 핸들을 닫고 Dino-Lite(a168:0960)를 sysfs에서 다시 찾는다.
USB 등록을 최대 10초 기다리며 영상 노드(index0)를 선택한다. /dev/video0가 video2 등으로
바뀌면 이후 영상 캡처와 제어 명령 모두 새 경로를 사용한다. 대상 장치가 없으면 시간초과,
동일 모델이 여러 대면 모호성 오류를 표시한다. 대기는 카메라 worker에서 수행한다.
터미널의 `Dino-Lite rediscovered: ... -> ...`를 확인한다. USB 재연결로 번호가 달라지는
경우까지 실물 검증이 필요하다. 최초 GUI 실행의 장치 경로 설정은 기존대로 유지한다.
