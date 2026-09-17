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
- 일반 측정: LED OFF → AE OFF → 동일한 노출 다섯 쓰기 → 대기 → 영상 속성 재적용 → 검사. AE ON을 반복하지 않는다.
- 실패: reset_flag_en=1이면 한 번 재연결하여 전체 초기화 후 재검사. 실패 지속 시 추론 차단.
- 대기 0.5초는 제공된 Python 코드 기준이다. 실제 PCAP 간격과 완전히 일치하지 않으며 40ms도 캡처의 35~38ms 간격을 근사한 값이다.
- 미확인 AXI/EFLC/렌즈 쓰기·내부 조회·장치 열기/포맷 협상은 복제하지 않는다. LED ON 패킷도 그대로 재생하지 않고 조명 OFF 목적을 유지한다. 정확한 영상 동등성은 실물 검증 대상이다.

## 사용자가 확인할 순서

1. Windows를 ExposureValue800/Brightness16 및 제공된 나머지 기본값으로 실행한다. 외부 조명·시료·배율·거리·ROI를 고정하고 RGB와 anomaly score를 5회 기록한다.
2. Jetson에서 ONNX backend로 앱을 재시작한다. 프로필 전환 비교는 재시작한다. 같은 물리 조건에서 5회 측정한다.
3. 터미널 Camera check 보고서의 camera_profile=windows_800, ExposureTime_requested=DNX64:800, video_controls readback 및 allow_inference를 확인한다. 노출 readback은 여전히 unavailable이다.
4. `v4l2-ctl -d /dev/video0 --list-ctrls-menus`에서 brightness16, contrast16, hue0, saturation32, sharpness0, gamma5, AWB0, WB5800을 확인한다. 자체 LED가 꺼져 있는지도 확인한다.
5. 연속 측정에서 RGB가 유지되는지 확인한다. 이상 광량을 넣어 복구를 발생시킨 뒤 광량을 원복하고, 정상 조건의 다음 측정이 원래 RGB로 돌아오는지 확인한다. 비정상 조명을 유지하면 FAIL이어야 한다.
6. 결과 공유: Windows/Jetson 각각 RGB 평균과 범위, score 평균과 범위, before/after 상태. BMP 반출은 필요 없다.

기존 모드 복귀: camera_profile을 fixed로 바꾸고 기존 ExposureTime과 video_controls/Brightness를 백업에서 복원한 뒤 재시작한다. 프로필 이름만 바꾸면 나머지 장치 설정까지 자동 원복되지는 않는다.

노출 시퀀스: 0502000c357810 → 0525000d357810 → 05000000357810 → 05320001357810 → 05000002357810. GUI 실행 중 별도 uvcdynctrl 수동 명령을 보내면 다음 측정에서 덮어쓰므로 비교 실험 중 병행하지 않는다.

## 미확인 쓰기 포함 실험: windows_800_full

camera_profile을 `windows_800_full`로 바꾸고 앱을 재시작한다. ExposureValue800, Brightness16, 위 Windows 기본 video_controls를 사용해야 한다. 캡처의 장치2:12에서 관측한 성공한 SET_CUR 쓰기 78개(XU unit4 및 PU unit3)를 원래 순서·반복대로 재생한다. 데이터는 scripts/windows_800_capture.json에 있다. 의미 미확인 쓰기 및 조회 준비용 e0 쓰기도 포함한다. LED ON 명령도 포함되므로 초기화 동안 LED가 일시적으로 켜질 수 있다.

USB 요청/응답을 통째로 복제하는 모드는 아니다. 장치 열기·포맷 협상·표준 USB 요청·GET 응답·영상은 제외한다. 따라서 e0 쓰기 뒤 원래 DLL이 수행한 GET까지 재현한 것은 아니다. 패킷 간 타임스탬프 차이를 sleep으로 사용하므로 실제 명령 실행시간이 더해져 Windows와 정확한 타이밍은 다르다. AWB는 캡처 조회 결과인 OFF를 재생 전에 명시적으로 설정한다. 재생 후 기존 고정 영상 속성을 재적용하고 readback·안정화한다.

일반 측정에서는 전체 78개를 반복하지 않고 AE OFF 및 노출 다섯 쓰기와 영상 속성만 재적용한다. 재연결 복구 때는 전체를 재생한다. 실패 시 정상 프레임으로 간주하지 않고 오류/복구 경로를 따른다. 프로필 전환은 반드시 재시작한다.

비교: windows_800에서 5회 RGB/score → windows_800_full로 재시작 후 동일 조건 5회 → 복구 후 5회. 자체 조명이 최종적으로 꺼졌는지 확인한다. 전체 옵션에서만 결과가 가까워지면 미확인 설정의 영향으로 범위를 좁힐 수 있지만 개별 명령 의미가 확정된 것은 아니다.
