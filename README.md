# Music Helper Drum - 드럼 사운드 분석 시스템

딥러닝을 활용하여 음원에서 드럼 사운드를 자동으로 감지하고 분류하는 AI 기반 시스템입니다.

## 주요 기능

- **9가지 드럼 클래스 자동 분류** (정확도 99%)
- **연속 음원에서 드럼 사운드 감지**
- **악보 스타일 타임라인 출력**
- **시간대별 드럼 패턴 분석**
- **시각화 및 상세 리포트 생성**

---

## 빠른 시작

### 필수 요구사항

- Python 3.10.x
- Conda (Anaconda 또는 Miniconda)

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/Music_Helper_Drum.git
cd Music_Helper_Drum

# 2. Conda 가상환경 생성 및 활성화
conda create -n MusicDrum python=3.10 -y
conda activate MusicDrum

# 3. 패키지 설치
pip install -r requirements.txt
```

### 실행 방법

```bash
# 1. 데이터 증강 (처음 한 번만)
python scripts/0_data_augmentation.py

# 2. 모델 학습 (처음 한 번만)
python models/1_model_training.py

# 3. 드럼 분석 실행
python models/4_score_analyzer.py
```

**더 자세한 내용은 [QUICK_START.md](QUICK_START.md)를 참고하세요.**

---

## 프로젝트 구조

```
Music_Helper_Drum/
├── scripts/                           # 데이터 전처리 및 테스트
│   ├── 0_data_augmentation.py        # 데이터 증강
│   ├── 0.1_data_preprocessing.py     # 기본 전처리
│   ├── 0.2_data_debugging.py         # 데이터 디버깅
│   ├── 3.1_test_continuous.py        # 연속 분석 테스트
│   └── 4.1_test_score.py             # 악보 분석 테스트
│
├── models/                            # 모델 및 분석기
│   ├── 1_model_training.py           # 모델 학습
│   ├── 2_model_predict.py            # 단일 드럼 예측
│   ├── 3_continuous_analyzer.py      # 연속 음원 분석
│   ├── 4_score_analyzer.py           # 악보 분석 (권장)
│   ├── best_crnn_model.h5            # 최적 모델
│   └── final_crnn_drum_model.h5      # 최종 모델
│
├── docs/                              # 상세 문서
│   ├── 9_class_info.md               # 드럼 클래스 정보
│   ├── DRUM_SCORE_ANALYSIS_GUIDE.md  # 악보 분석 가이드
│   └── ANALYSIS_COMPARISON.md        # 분석 방법 비교
│
├── drum_samples/                      # 학습용 드럼 샘플
├── TestSound/                         # 테스트 오디오 파일
├── output/                            # 분석 결과 출력
├── data_augmented/                    # 증강된 학습 데이터
├── debug_img/                         # 디버깅 이미지
│
├── README.md                          # 프로젝트 소개 (이 파일)
├── QUICK_START.md                     # 빠른 시작 가이드
└── PROJECT_WORKFLOW.md                # 상세 워크플로우 가이드
```

---

## 지원하는 드럼 클래스 (9개)

| 약어 | 영문명 | 한국어명 |
|------|--------|----------|
| BD | bass_drum | 베이스드럼 |
| SD | snare_drum | 스네어드럼 |
| HT | high_tom | 하이톰 |
| MT | mid_tom | 미드톰 |
| LT | low_tom | 로우톰 |
| HH | hihat_closed | 클로즈드하이햇 |
| OH | hihat_open | 오픈하이햇 |
| CC | crash_cymbal | 크래시심벌 |
| RC | ride_cymbal | 라이드심벌 |

---

## 사용 예시

### 1. 단일 드럼 사운드 분류

```python
from models.2_model_predict import DrumPredictor

predictor = DrumPredictor("models/final_crnn_drum_model.h5")
results = predictor.predict("TestSound/crash.wav")

for result in results:
    print(f"{result['class']}: {result['confidence']}")
```

### 2. 연속 음원 악보 분석 (권장)

```python
from models.4_score_analyzer import DrumScoreAnalyzer

# 분석기 초기화
analyzer = DrumScoreAnalyzer("models/final_crnn_drum_model.h5")

# 음원 분석
results, duration = analyzer.analyze_continuous_audio("your_audio.wav")

# 결과 출력
analyzer.print_score_timeline(results, duration)

# 시각화 저장
analyzer.create_score_visualization(results, duration, "your_audio.wav", 
                                   save_path="output/result.png")
```

### 3. TestSound 폴더 활용

**TestSound 폴더에 분석할 오디오 파일을 넣고 실행하면 자동으로 모든 파일을 분석합니다.**

```bash
# TestSound 폴더에 오디오 파일 추가 후
python models/4_score_analyzer.py
```

결과:
- `output/파일명_analysis.png` - 시각화
- `output/파일명_analysis.txt` - 상세 결과

---

## 출력 예시

```
================================================================================
드럼 악보 분석 결과
================================================================================

[분석 통계]
총 음원 길이: 2.00초
감지된 드럼 사운드: 3개

[클래스별 분포]
  CC (크래시심벌): 1개
  MT (미드톰): 1개
  LT (로우톰): 1개

[타임라인 - 악보 형식]
--------------------------------------------------------------------------------
시간(초)   | 드럼 종류            | 신뢰도    
--------------------------------------------------------------------------------
  0.00     | CC (크래시심벌)      | 91.5%     
  0.16     | MT (미드톰)          | 100.0%    
  0.42     | LT (로우톰)          | 100.0%    
```

---

## 모델 아키텍처

**CRNN (Convolutional Recurrent Neural Network)**

- **CNN 부분**: 특징 추출 (Conv2D 32→64→128→256)
- **RNN 부분**: 시간적 패턴 학습 (LSTM 128→64)
- **정확도**: 약 99%
- **입력**: 멜 스펙트로그램 (128 mel bins)
- **샘플레이트**: 22050Hz

---

## 문서

### 필수 문서
- **[QUICK_START.md](QUICK_START.md)** - 빠른 시작 가이드
- **[PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md)** - 상세 워크플로우 및 코드 실행 순서

### 상세 문서 (docs/)
- **[9_class_info.md](docs/9_class_info.md)** - 9가지 드럼 클래스 정보
- **[DRUM_SCORE_ANALYSIS_GUIDE.md](docs/DRUM_SCORE_ANALYSIS_GUIDE.md)** - 악보 분석 시스템 가이드
- **[ANALYSIS_COMPARISON.md](docs/ANALYSIS_COMPARISON.md)** - 세 가지 분석 방법 비교

---

## 지원 오디오 형식

- WAV (권장)
- MP3
- FLAC
- OGG
- AIF/AIFF

---

## 기술 스택

- **딥러닝**: TensorFlow/Keras
- **오디오 처리**: Librosa
- **데이터 분석**: NumPy, Pandas
- **시각화**: Matplotlib, Seaborn
- **머신러닝**: Scikit-learn

---

## 활용 예시

- **드럼 커버 연습** - 원곡 드럼 패턴 분석
- **음악 제작** - 드럼 패턴 참고
- **음악 교육** - 드럼 악보 자동 생성
- **음원 분석 연구**

---

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

---

## 감사의 말

이 프로젝트는 다음 오픈소스 라이브러리들을 사용합니다:
- Librosa - 오디오 분석
- TensorFlow/Keras - 딥러닝
- Scikit-learn - 머신러닝
- NumPy, Pandas - 데이터 처리
- Matplotlib, Seaborn - 시각화

---

**Made for Musicians and Audio Engineers**
