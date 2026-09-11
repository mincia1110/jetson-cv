# 2026-09-11: VisualAD 후처리 연결 완료

사용자가 제공한 VisualADInferencer와 공식 저장소의 scoring/get_transform 정의를 바탕으로 `scripts/visualad_adapter.py`를 추가했다. 기존 템플릿의 전후처리 미구현 안내는 다른 모델용이며, 이번 7-output VisualAD 모델에는 이 어댑터를 사용한다. 사내 get_transform이 현재 공개 버전과 동일한지는 실제 입력 텐서로 최종 비교한다.

기존 local/measurement.json의 adapter를 `scripts/visualad_adapter.py`로 변경한다. 예제 설정도 이 경로로 갱신했다. 입력은 float32 [1,3,336,336], input 이름이다. class_features는 원본 wrapper처럼 사용하지 않는다. 4개 레이어 모두 존재해야 한다.

의존성은 torch와 scipy다. 모델 실행은 ONNX CUDA 또는 TensorRT GPU이고, 제공된 wrapper와 동일하게 후처리는 CPU torch/scipy에서 실행한다. 사용 중인 Jetson용 torch를 유지한다.

```bash
python -c "import torch, scipy; print(torch.__version__, scipy.__version__)"
python -m pip install -r requirements-inference.txt
```

Torch가 없으면 해당 환경용 wheel 설치가 별도로 필요하다. NumPy/OpenCV 버전이나 NVIDIA torch를 무조건 업그레이드하지 않는다.

## 이미 저장한 출력 비교 (이미지 없이 가능)

아래 threshold 0은 실행 예시다. 실제 GUI threshold를 10으로 나눈 내부값으로 바꾼다.

```bash
python scripts/compare_visualad.py \
  --reference artifacts/backend-comparison-01/onnx_outputs.npz \
  --actual artifacts/backend-comparison-01/trt_outputs.npz \
  --threshold 0 \
  --output artifacts/visualad-postprocess-01
```

합산 raw map의 상위 1%(ceil, 1129개) 평균이 score다. sigma=4 Gaussian은 그 이후 맵·마스크에만 적용한다. wrapper label/mask는 strict >, GUI의 NG는 >=를 유지한다. 출력은 맵·점수 오차, 양쪽 판정, 마스크 불일치 픽셀 수다. 0 입력 결과는 실행·수치 비교용이며 업무 판정 검증은 사내 실제 영상으로 한다.

출처: [공식 scoring](https://github.com/7HHHHH/VisualAD/blob/main/utils/scoring.py), [공식 transforms](https://github.com/7HHHHH/VisualAD/blob/main/utils/transforms.py). 사용자 제공 wrapper의 계산 순서를 기준으로 구현했으며 아래의 과거 템플릿 연결 설명보다 이 절차를 우선한다.

---

# test.py → ONNX Runtime → TensorRT

`python scripts/test_jetson.py --config local/measurement.json`이 측정 GUI 진입점이다. 제공받은 test.py의 Tkinter 배치(미리보기, 전처리/NG/히트맵, 이름·폴더·threshold·측정·OK/NG)를 기준으로 카메라와 추론 호출을 분리했다. 원본 파일 자체는 저장소에 없으므로 원본 전체에 대한 패치가 아니라 제공된 코드에서 측정 경로를 옮긴 버전이다.

## 유지·변경 범위

- 카메라: 기존 DLL/DirectShow 대신 DinoLiteCamera. 초기 설정과 추론 전 상태 검사, 조건 통과한 동일 프레임 전달.
- 명시된 전처리: 원본 해상도, ROI, blur, 전체 min-max, gamma, 336 패딩 유지. 마지막 get_transform은 원본 구현이 필요하다.
- 판정: UI threshold -20~20을 /10, score < threshold이면 OK. 히트맵은 OpenCV로 표시하며 기존 Matplotlib 출력과 픽셀 단위 외관은 다르다.
- 기존 실험용 5회 반복 대신 버튼당 1회. 시간 표시된 폴더에 저장해 덮어쓰기를 방지한다. CSV/column500 실험과 주석 처리된 Subspace 융합은 이 버전에 옮기지 않았다.
- 파일 입력은 `--file-mode`. 카메라 검사를 수행했다고 표시하지 않는다.
- 추론 스레드 하나가 CUDA 세션 생성·실행·종료를 담당한다. Tk 위젯은 메인 스레드에서만 갱신한다.

## 설치와 실제 모델 연결

카메라 설치는 CAMERA_CONTROL.md를 따른다. 추가 패키지:

```bash
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
mkdir -p local
cp examples/measurement.example.json local/measurement.json
cp examples/model_adapter.py local/model_adapter.py
```

ONNX Runtime은 **Jetson aarch64/Python/CUDA/cuDNN에 맞는 CUDA wheel**을 별도로 준비한다. 준비한 wheel은 `python -m pip install /path/to/validated.whl`로 설치한다. 최신 onnxruntime-gpu를 무조건 설치하지 않는다. CUDA/cuDNN의 호환성은 [공식 표](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), 직접 빌드는 [공식 안내](https://onnxruntime.ai/docs/build/eps.html)를 따른다. 이 저장소는 특정 Jetson용 wheel을 검증·동봉하지 않는다.

```bash
python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

CUDAExecutionProvider가 필요하다. 런타임은 CPU fallback을 비활성화한다. CUDA 미지원 노드가 있으면 오류를 내며 임의로 GPU 전용 조건을 완화하지 않는다. CPU 비교가 목적일 때만 설정의 provider를 CPUExecutionProvider로 명시한다.

`local/model_adapter.py`의 두 함수를 사내 원본 코드로 채운다.

- preprocess(frame_bgr, config): 원본 get_transform 포함, 실제 ONNX 이름/shape/dtype의 feed dict와 결과 맵에 정렬된 uint8 RGB 표시 이미지 반환.
- postprocess(outputs, threshold, config): 기존 user_th_inference의 출력 해석·스케일링을 그대로 옮긴다. 2D anomaly_map과 scalar pred_score 반환.

**이 두 모델별 함수는 자료 부재로 자동 완성할 수 없어서 예제는 명시적으로 오류를 낸다.** 입력 이름, 정규화, raw feature를 anomaly score로 추정하지 않는다. 별도 examples/map_score_adapter.py는 최종 336×336 map과 단일 score를 직접 출력하는 ONNX에만 사용할 수 있다. input_name/input_mean/input_std/map_output/score_output을 설정에 명시해야 하며 실제 VisualAD에 맞는다고 검증된 어댑터가 아니다.

설정 경로는 프로젝트 루트 기준이다. 모델별 JSON을 만들어 --config로 선택한다. 기본 CUDA 모델 경로는 onnx 키, TensorRT는 engine 키다. 카메라 ExposureValue=800 대응은 별도 미해결이며 CAMERA_CONTROL.md의 범위를 따른다.

## 1. ONNX 입력 확인·실행

```bash
python scripts/onnx_probe.py --model local/model.onnx --output artifacts/onnx-io
python scripts/test_jetson.py --config local/measurement.json --file-mode
```

실제 어댑터를 연결한 뒤 GUI에서 BMP, 저장 폴더, 이름을 선택하고 측정한다. 각 측정 폴더에 original.bmp, inputs.npz, outputs.npz, anomaly_map.npy, report.json, result.jpg가 저장된다. 파일 모드로 확인 후 --file-mode를 빼면 카메라로 실행한다.

저장한 텐서만으로 ONNX를 다시 실행할 수도 있다.

```bash
python scripts/onnx_probe.py --model local/model.onnx --inputs artifacts/measurement/inputs.npz --output artifacts/onnx-check
```

## 2. TensorRT 변환

기존 변환기를 그대로 사용한다. 대상 Jetson에서 수행하며 매번 새 출력 폴더를 쓴다.

```bash
bash scripts/build_engine.sh local/model.onnx artifacts/visualad-fp32
```

동적 입력이면 실제 입력 이름과 shape에 맞춘 --minShapes/--optShapes/--maxShapes를 추가한다. 변환기는 GPU용 engine, 명령, 로그, 해시를 저장한다. 실제 모델의 지원 연산과 플러그인은 대상에서 확인한다.

## 3. 동일 입력 비교·GUI 전환

```bash
python scripts/compare_backends.py --onnx local/model.onnx --engine artifacts/visualad-fp32/model.engine --inputs artifacts/measurement/inputs.npz --output artifacts/backend-comparison
```

이름·shape·유한값·수치 오차를 검사한다. 비교 실패 시 종료 코드 1이다. 이 도구는 raw outputs 비교이며 업무 판정 일치 자체를 보장하지 않는다. 허용오차는 모델별로 정하고 threshold 근처 사례도 GUI에서 비교한다.

`local/measurement.json`의 backend를 tensorrt로 바꿔 같은 BMP로 다시 측정한다. 같은 어댑터가 사용된다. FP32 일치 확인 후 필요하면 별도 FP16 엔진을 만들어 동일 절차로 비교한다.

## 검증 한계

로컬 모의 테스트와 합성 ONNX CPU 실행은 코드 경로 확인이다. 실제 VisualAD 모델/사내 전후처리, Jetson ONNX CUDA wheel, TensorRT 엔진 실행, 실물 카메라 노출 호환성, Tk 화면의 실물 QA는 별도 대상 검증이 필요하다. 이 항목을 완료했다고 간주하지 않는다.
