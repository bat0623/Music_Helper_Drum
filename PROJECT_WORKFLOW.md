# Music Helper Drum - 프로젝트 워크플로우 가이드

이 문서는 Music Helper Drum 프로젝트의 코드 실행 순서와 각 파일의 역할을 정리한 가이드입니다.

## 프로젝트 구조

```
Music_Helper_Drum/
├── scripts/              # 데이터 전처리 및 테스트 스크립트
│   ├── 0_data_augmentation.py          # 데이터 증강
│   ├── 0.1_data_preprocessing.py       # 기본 데이터 전처리
│   ├── 0.2_data_debugging.py          # 데이터 디버깅 도구
│   ├── 3.1_test_continuous.py         # 연속 음원 분석 테스트
│   └── 4.1_test_score.py              # 악보 분석 테스트
│
├── models/               # 모델 및 분석기
│   ├── 1_model_training.py            # 모델 학습
│   ├── 2_model_predict.py             # 단일 드럼 예측
│   ├── 3_continuous_analyzer.py       # 연속 음원 분석기
│   ├── 4_score_analyzer.py            # 악보 분석기
│   ├── best_crnn_model.h5             # 최적 모델
│   └── final_crnn_drum_model.h5       # 최종 모델
│
├── TestSound/            # 테스트 오디오 파일
├── output/              # 분석 결과 출력 폴더
└── debug_img/           # 디버깅 이미지
```

---

## 코드 실행 순서

### 단계 0: 데이터 준비 및 전처리

#### 0.1 기본 데이터 전처리 (처음 사용 시)
**파일**: `scripts/0.1_data_preprocessing.py`

```bash
python scripts/0.1_data_preprocessing.py
```

**역할**:
- 원본 드럼 샘플 데이터 로딩 (`drum_samples/` 폴더)
- 멜 스펙트로그램 변환
- 전처리된 데이터를 `.npy` 파일로 저장 (`data/` 폴더)

**입력**: `drum_samples/` 폴더의 9개 클래스별 `.wav` 파일들
**출력**: `data/X_data.npy`, `data/y_data.npy`

---

#### 0 데이터 증강 (성능 향상을 위해)
**파일**: `scripts/0_data_augmentation.py`

```bash
python scripts/0_data_augmentation.py
```

**역할**:
- 시간 축 이동(Time Shift) 방식으로 데이터 증강
- 원본 데이터 1개당 100개의 증강 데이터 생성
- 증강된 데이터를 `.npy` 파일로 저장

**입력**: `drum_samples/` 폴더
**출력**: `data_augmented/X_data.npy`, `data_augmented/y_data.npy`

**주요 설정**:
- `AUGMENTATIONS_PER_FILE = 100`: 파일당 증강 개수
- `TIME_SHIFT_SECONDS = 0.2`: 최대 시간 이동 범위

---

#### 0.2 데이터 디버깅 (선택사항)
**파일**: `scripts/0.2_data_debugging.py`

```bash
python scripts/0.2_data_debugging.py
```

**역할**:
- 전처리된 `.npy` 데이터 검증
- 각 클래스별 멜 스펙트로그램 시각화
- 데이터 형태 및 내용 확인

---

### 단계 1: 모델 학습

**파일**: `models/1_model_training.py`

```bash
python models/1_model_training.py
```

**역할**:
- CRNN(CNN + LSTM) 모델 구축
- 증강된 데이터로 모델 학습
- 훈련/검증/테스트 세트 분할 (70%/10%/20%)
- 모델 평가 및 혼동 행렬 생성

**주요 클래스**:
- `DrumDataLoader`: 데이터 로딩 및 분할
- `CRNNDrumClassifier`: CRNN 모델 아키텍처
- `ModelTrainer`: 모델 훈련
- `ModelEvaluator`: 모델 평가

**입력**: `data_augmented/X_data.npy`, `data_augmented/y_data.npy`
**출력**:
- `models/best_crnn_model.h5`: 검증 정확도가 가장 높은 모델
- `models/final_crnn_drum_model.h5`: 최종 학습 완료 모델
- `debug_img/training_history.png`: 훈련 과정 그래프
- `debug_img/confusion_matrix.png`: 혼동 행렬

**학습 설정**:
- Epochs: 100 (Early Stopping 적용)
- Batch Size: 32
- Optimizer: Adam (learning rate: 0.001)

---

### 단계 2: 모델 예측 (단일 드럼 사운드)

**파일**: `models/2_model_predict.py`

```bash
python models/2_model_predict.py
```

**역할**:
- 학습된 모델로 단일 드럼 사운드 예측
- TestSound 폴더의 모든 오디오 파일 자동 분석
- 상위 3개 예측 결과 출력

**주요 클래스**:
- `DrumPredictor`: 드럼 사운드 예측기

**입력**: `TestSound/` 폴더의 오디오 파일들
**출력**: 콘솔에 예측 결과 출력

**사용 예시**:
```python
# 데모 실행 (TestSound 폴더 자동 분석)
demo_prediction()

# 대화형 예측
interactive_prediction()
```

---

### 단계 3: 연속 음원 분석

**파일**: `models/3_continuous_analyzer.py`

```bash
python models/3_continuous_analyzer.py
```

**역할**:
- 연속된 드럼 연주 음원 분석
- 슬라이딩 윈도우 또는 온셋 기반 분석
- 타임라인 시각화 및 결과 내보내기

**주요 클래스**:
- `ContinuousDrumAnalyzer`: 연속 음원 분석기

**분석 방법**:
1. **슬라이딩 윈도우 (Sliding Window)**
   - 고정된 윈도우를 일정 간격으로 이동하며 분석
   - 윈도우 크기: 2.0초
   - 이동 간격: 0.5초

2. **온셋 기반 (Onset-based)**
   - 드럼 사운드 시작점을 자동 감지하여 분석
   - 더 정확하지만 감지 실패 가능성 있음

**입력**: `TestSound/` 폴더의 오디오 파일들
**출력**:
- `output/[파일명]_continuous_sw.png`: 슬라이딩 윈도우 분석 시각화
- `output/[파일명]_continuous_sw.txt`: 분석 결과 텍스트

**주요 설정**:
- `window_size = 2.0`: 윈도우 크기 (초)
- `hop_size = 0.5`: 윈도우 이동 간격 (초)
- `confidence_threshold = 0.7`: 신뢰도 임계값

---

#### 3.1 연속 음원 분석 테스트

**파일**: `scripts/3.1_test_continuous.py`

```bash
python scripts/3.1_test_continuous.py
```

**역할**:
- 연속 음원 분석기의 두 가지 방법 비교 테스트
- 다양한 파라미터 조정 테스트
- 결과 비교 및 최적 설정 찾기

---

### 단계 4: 악보 분석

**파일**: `models/4_score_analyzer.py`

```bash
python models/4_score_analyzer.py
```

**역할**:
- 연속 음원을 악보 형식으로 분석
- 더 짧은 윈도우(0.5초)로 정밀 분석
- 타임라인, 클래스별 통계, 상세 결과 제공

**주요 클래스**:
- `DrumScoreAnalyzer`: 악보 분석기

**출력 형식**:
1. **타임라인**: 시간별 드럼 사운드 발생 목록
2. **클래스별 분포**: 각 드럼 종류별 개수
3. **상세 결과**: 각 사운드의 정확한 시간 및 신뢰도
4. **시각화**: 악보 스타일 타임라인 그래프

**입력**: `TestSound/` 폴더의 오디오 파일들
**출력**:
- `output/[파일명]_analysis.png`: 악보 스타일 시각화
- `output/[파일명]_analysis.txt`: 상세 분석 결과

**주요 설정**:
- `window_size = 0.5`: 윈도우 크기 (초)
- `hop_size = 0.05`: 윈도우 이동 간격 (초)
- `confidence_threshold = 0.5`: 신뢰도 임계값

**드럼 약어 (Abbreviations)**:
- BD: 베이스드럼 (Bass Drum)
- SD: 스네어드럼 (Snare Drum)
- HT: 하이톰 (High Tom)
- MT: 미드톰 (Mid Tom)
- LT: 로우톰 (Low Tom)
- HH: 클로즈드하이햇 (Hihat Closed)
- OH: 오픈하이햇 (Hihat Open)
- CC: 크래시심벌 (Crash Cymbal)
- RC: 라이드심벌 (Ride Cymbal)

---

#### 4.1 악보 분석 테스트

**파일**: `scripts/4.1_test_score.py`

```bash
python scripts/4.1_test_score.py
```

**역할**:
- 악보 분석기 테스트
- TestSound 폴더의 모든 파일 자동 분석
- 결과를 output 폴더에 저장

---

## 드럼 클래스 (9개)

| 번호 | 영문명 | 한글명 | 약어 |
|------|--------|--------|------|
| 0 | bass_drum | 베이스드럼 | BD |
| 1 | crash_cymbal | 크래시심벌 | CC |
| 2 | high_tom | 하이톰 | HT |
| 3 | hihat_closed | 클로즈드하이햇 | HH |
| 4 | hihat_open | 오픈하이햇 | OH |
| 5 | low_tom | 로우톰 | LT |
| 6 | mid_tom | 미드톰 | MT |
| 7 | ride_cymbal | 라이드심벌 | RC |
| 8 | snare_drum | 스네어드럼 | SD |

---

## 파라미터 조정 가이드

### 데이터 증강 (`scripts/0_data_augmentation.py`)
```python
AUGMENTATIONS_PER_FILE = 100  # 파일당 증강 개수 (더 많을수록 데이터 증가)
TIME_SHIFT_SECONDS = 0.2      # 시간 이동 범위 (초)
```

### 연속 음원 분석 (`models/3_continuous_analyzer.py`)
```python
window_size = 2.0              # 윈도우 크기 (초) - 클수록 문맥 인식 향상
hop_size = 0.5                 # 이동 간격 (초) - 작을수록 정밀하지만 느림
confidence_threshold = 0.7     # 신뢰도 임계값 - 높을수록 정확하지만 놓칠 수 있음
```

### 악보 분석 (`models/4_score_analyzer.py`)
```python
window_size = 0.5              # 윈도우 크기 (초)
hop_size = 0.05                # 이동 간격 (초) - 매우 정밀한 분석
confidence_threshold = 0.5     # 신뢰도 임계값
```

---

## 사용 팁

1. **처음 사용 시 순서**:
   ```bash
   python scripts/0_data_augmentation.py      # 데이터 증강
   python models/1_model_training.py          # 모델 학습
   python models/2_model_predict.py           # 예측 테스트
   python models/3_continuous_analyzer.py     # 연속 음원 분석
   python models/4_score_analyzer.py          # 악보 분석
   ```

2. **이미 학습된 모델이 있는 경우**:
   - 1단계(모델 학습)는 건너뛰고 2~4단계만 실행

3. **새로운 드럼 샘플 추가 시**:
   - `drum_samples/` 폴더에 클래스별로 추가
   - 0단계부터 다시 실행

4. **성능 향상 방법**:
   - 데이터 증강 개수 늘리기 (`AUGMENTATIONS_PER_FILE`)
   - 더 많은 원본 샘플 수집
   - 모델 학습 epoch 조정

5. **분석 결과가 부정확한 경우**:
   - 신뢰도 임계값 조정
   - 윈도우 크기 조정
   - 온셋 기반 분석 시도

---

## 주요 폴더

- **drum_samples/**: 원본 드럼 샘플 데이터
- **data/**: 전처리된 데이터 (기본)
- **data_augmented/**: 증강된 데이터
- **TestSound/**: 테스트용 오디오 파일
- **output/**: 분석 결과 출력 폴더
- **debug_img/**: 모델 훈련 결과 이미지
- **models/**: 학습된 모델 파일 및 분석 코드
- **scripts/**: 데이터 처리 및 테스트 스크립트

---

## 모델 아키텍처

**CRNN (Convolutional Recurrent Neural Network)**

1. **CNN 부분** (특징 추출):
   - Conv2D 32 → Conv2D 64 → Conv2D 128 → Conv2D 256
   - 각 층마다 BatchNormalization, MaxPooling, Dropout 적용

2. **RNN 부분** (시간적 패턴 학습):
   - LSTM 128 → LSTM 64
   - Dropout 적용

3. **완전연결층** (분류):
   - Dense 128 → Dense 64 → Dense 9 (Softmax)

---

## 시스템 요구사항

- Python 3.7+
- TensorFlow 2.x
- Librosa
- NumPy
- Matplotlib
- Scikit-learn
- tqdm

설치 방법:
```bash
pip install -r requirements.txt
```

---

## 문제 해결

1. **모델 로딩 실패**:
   - `models/final_crnn_drum_model.h5` 파일 존재 확인
   - 먼저 `1_model_training.py` 실행 필요

2. **데이터 로딩 실패**:
   - `.npy` 파일 존재 확인
   - `0_data_augmentation.py` 또는 `0.1_data_preprocessing.py` 실행 필요

3. **메모리 부족**:
   - Batch size 줄이기
   - 데이터 증강 개수 줄이기

4. **느린 실행 속도**:
   - GPU 사용 확인 (TensorFlow GPU 버전)
   - 윈도우 이동 간격 늘리기

---

**작성일**: 2025-10-23
**버전**: 1.0
