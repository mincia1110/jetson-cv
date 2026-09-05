# 로컬 검증 기록

2026-09-05, macOS ARM64 / Python 3.9.6. Jetson, Dino-Lite, 사내 ONNX 미연결.
목표는 ONNX를 대상 Jetson에서 TensorRT 엔진으로 변환한 뒤 기존 PyQt 앱에서 실행하는 것이다.

테스트 의존성은 `/tmp/jetson-cv-check` 가상환경에 설치했다: NumPy 1.26.4, ONNX 1.16.2, Polygraphy 0.49.9. 전역 패키지는 변경하지 않았다. 최초 다운로드는 샌드박스 DNS 제한으로 실패했으며 승인된 네트워크 재시도로 설치했다.

| 명령 | 결과 |
| --- | --- |
| `/tmp/jetson-cv-check/bin/python -m unittest discover -s tests -v` | 6개 통과 |
| `/tmp/jetson-cv-check/bin/python scripts/make_smoke_model.py --output /tmp/jetson-cv-smoke.onnx` | ONNX checker 통과, 모델/입력/기준 출력 생성 |
| `PYTHONPYCACHEPREFIX=/tmp/jetson-cv-pycache python3 -m compileall -q scripts tests` | 종료 코드 0 |
| `bash -n scripts/build_engine.sh scripts/setup_jetson.sh scripts/trt_smoke.sh` | 종료 코드 0 |
| `/tmp/jetson-cv-check/bin/python scripts/trt_probe.py /tmp/missing.engine --inputs /tmp/jetson-cv-smoke.inputs.npz --output /tmp/jetson-cv-trt-missing-runtime` | 예상대로 코드 1, TensorRT 없음과 pass=false 보고 |

6개 테스트: 오차 범위 초과, broadcast 가능한 shape 불일치, 이름 불일치, NaN/Inf/빈 출력 거부; 입력 dtype와 scalar shape 보존; 반복 추론 버퍼에서 출력 복사; 비활성 세션 거부; 가짜 trtexec의 실패 코드가 tee에 가려지지 않고 이전 엔진을 덮어쓰지 않는지 확인했다. 가짜 runner/builder 테스트는 실제 GPU 실행 증거가 아니다.

추가로 ONNX 공식 ReferenceEvaluator에 생성된 inputs.npz를 전달하고 `np.testing.assert_array_equal`로 expected.npz와 정확히 일치함을 확인했다. 설치된 Polygraphy 0.49.9 소스에서 TrtRunner의 infer/check_inputs/copy_outputs_to_host, execute_async_v3 및 stream 동기화 경로를 확인했다.

이전 단계의 환경 보호 검사: `bash scripts/setup_jetson.sh`는 Mac에서 설치 전에 코드 1로 중단했다. Python 기본 캐시 쓰기 제한은 /tmp 캐시 지정으로 해결했다.

미실행: JetPack 설치, Jetson apt/pip 설치, 실제 TensorRT 엔진 빌드/역직렬화/GPU 실행, 실제 카메라 획득, PyQt 앱 이식, 실제 모델 정확도/메모리/속도, 오프라인 재부팅 검증. README의 장비 인수 절차로 수행해야 한다. ONNX Runtime 점검 스크립트와 관련 테스트는 목표 정정에 따라 제거했다.

초기 개발 시 작업 폴더는 Git 저장소가 아니었다. 이후 사용자 요청으로 GitHub 프라이빗 저장소 반입용 README를 보완하고 Git 저장소를 초기화했다. 게시 전 위 6개 테스트, compileall, bash -n을 재실행하여 모두 통과했고 `git diff --cached --check`도 통과했다. 실제 게시 상태는 GitHub 저장소와 Git 이력에서 확인한다.
