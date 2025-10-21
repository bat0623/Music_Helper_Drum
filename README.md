# Music Helper Drum - 드럼 사운드 분석 시스템

## 프로젝트 소개

딥러닝을 활용하여 음원에서 드럼 사운드를 자동으로 감지하고 분류하는 AI 기반 시스템입니다.

### 주요 기능

- 9가지 드럼 클래스 자동 분류 (정확도 99%)
- 연속 음원에서 드럼 사운드 감지
- 악보 스타일 타임라인 출력
- 시간대별 드럼 패턴 분석
- 시각화 및 상세 리포트 생성

## 시작하기

### 필수 요구사항

- Python 3.10.x
- Conda (Anaconda 또는 Miniconda)

### 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/yourusername/Music_Helper_Drum.git
cd Music_Helper_Drum
```

2. **Conda 가상환경 생성 및 활성화**
```bash
conda create -n MusicDrum python=3.10 -y
conda activate MusicDrum
```

3. **패키지 설치**
```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
Music_Helper_Drum/
├── models/                         # 모델 및 분석 시스템
│   ├── crnn_drum_classifier.py    # 모델 학습 스크립트
│   ├── final_crnn_drum_model.h5   # 학습된 모델 파일
│   ├── predict_drum.py            # 단일 드럼 분류
│   ├── continuous_drum_analyzer.py # 연속 음원 분석
│   └── drum_score_analyzer.py     # 악보 스타일 분석 (권장)
│
├── scripts/                        # 스크립트 및 도구
│   ├── Data augmentation (time axis shift).py  # 데이터 증강
│   ├── npy_debugging.py           # 데이터 디버깅
│   ├── test.py                    # 테스트 스크립트
│   ├── test_continuous_analysis.py # 연속 분석 테스트
│   └── test_score_analysis.py     # 악보 분석 테스트
│
├── drum_samples/                   # 드럼 샘플 데이터 (학습용)
│   ├── bass_drum/                 # 베이스드럼
│   ├── crash_cymbal/              # 크래시심벌
│   ├── high_tom/                  # 하이톰
│   ├── hihat_closed/              # 클로즈드하이햇
│   ├── hihat_open/                # 오픈하이햇
│   ├── low_tom/                   # 로우톰
│   ├── mid_tom/                   # 미드톰
│   ├── ride_cymbal/               # 라이드심벌
│   └── snare_drum/                # 스네어드럼
│
├── TestSound/                     # 테스트용 오디오 파일
│   ├── Crash4.wav
│   ├── TomHi4.wav
│   ├── Untitled.wav
│   └── final.wav
│
├── data/                          # 원본 데이터 (npy 파일)
├── data_augmented/                # 증강된 데이터 (npy 파일)
├── output/                        # 분석 결과 출력
│
├── DRUM_SCORE_ANALYSIS_GUIDE.md   # 악보 분석 가이드
├── ANALYSIS_COMPARISON.md         # 분석 방법 비교
└── requirements.txt               # 패키지 의존성
```

## 사용 방법

### 1. 단일 드럼 사운드 분류

개별 드럼 사운드 파일을 분류합니다.

```python
from models.predict_drum import DrumPredictor

predictor = DrumPredictor("models/final_crnn_drum_model.h5")
results = predictor.predict("drum_samples/snare_drum/snare1.wav")

for result in results:
    print(f"{result['class']}: {result['confidence']}")
```

### 2. 연속 음원 분석 (권장)

연속된 음원에서 드럼 사운드를 감지하고 악보처럼 표시합니다.

```python
from models.drum_score_analyzer import DrumScoreAnalyzer

# 분석기 초기화
analyzer = DrumScoreAnalyzer("models/final_crnn_drum_model.h5")

# 음원 분석
results, duration = analyzer.analyze_continuous_audio("your_audio.wav")

# 결과 출력 (악보 스타일)
analyzer.print_score_timeline(results, duration)

# 시각화
analyzer.create_score_visualization(results, duration, "your_audio.wav", 
                                   save_path="output/result.png")

# 결과 저장
analyzer.export_results(results, duration, "output/result.txt")
```

### 3. TestSound 폴더 활용

**TestSound 폴더에 분석할 오디오 파일을 넣으면 자동으로 모든 파일을 분석합니다.**

```bash
# 1. TestSound 폴더에 오디오 파일 추가
# 예: final.wav, Untitled.wav, Crash4.wav 등

# 2. 테스트 스크립트 실행 (자동으로 모든 파일 분석)
python scripts/test_score_analysis.py

# 또는 직접 실행
python models/drum_score_analyzer.py
```

**출력 결과:**
- 각 파일마다 개별 분석 결과 출력
- `output/파일명_analysis.png` - 시각화 결과
- `output/파일명_analysis.txt` - 상세 분석 결과

### 4. 기타 테스트

```bash
# 연속 음원 분석 테스트 (슬라이딩 윈도우 + 온셋 기반)
python scripts/test_continuous_analysis.py

# 단일 드럼 분류 테스트
python models/predict_drum.py
```

## 분석 시스템 비교

| 시스템 | 용도 | 특징 |
|--------|------|------|
| **predict_drum.py** | 단일 드럼 분류 | 빠른 속도, 단일 파일 처리 |
| **continuous_drum_analyzer.py** | 연속 음원 분석 | 슬라이딩 윈도우/온셋 기반 |
| **drum_score_analyzer.py** | 악보 스타일 분석 | 정밀 분석, 악보 출력 (권장) |

상세 비교는 [ANALYSIS_COMPARISON.md](ANALYSIS_COMPARISON.md) 참고

## 출력 예시

### 콘솔 출력
```
================================================================================
드럼 악보 분석 결과
================================================================================

[분석 통계]
총 음원 길이: 2.00초
감지된 드럼 사운드: 3개

[클래스별 분포]
  CC (크래시심벌): 1개
  LT (로우톰): 1개
  MT (미드톰): 1개

[타임라인 - 악보 형식]
--------------------------------------------------------------------------------
시간(초)   | 드럼 종류            | 신뢰도    
--------------------------------------------------------------------------------
  0.00     | CC (크래시심벌)      | 91.5%     
  0.16     | MT (미드톰)          | 100.0%    
  0.42     | LT (로우톰)          | 100.0%    

[상세 타임라인 - 0.1초 단위]
--------------------------------------------------------------------------------
 0.00s | CC
 0.10s | MT
 0.20s | MT
 0.40s | LT
 0.50s | LT
```

### 시각화 출력
- 악보 스타일 타임라인 차트
- 클래스별 분포 그래프
- 시간대별 밀도 분석

## 지원하는 드럼 클래스

| 약어 | 영문명 | 한국어명 |
|------|--------|----------|
| BD | bass_drum | 베이스드럼 |
| CC | crash_cymbal | 크래시심벌 |
| HT | high_tom | 하이톰 |
| HH | hihat_closed | 클로즈드하이햇 |
| OH | hihat_open | 오픈하이햇 |
| LT | low_tom | 로우톰 |
| MT | mid_tom | 미드톰 |
| RC | ride_cymbal | 라이드심벌 |
| SD | snare_drum | 스네어드럼 |

## 모델 성능

- **모델**: CRNN (Convolutional Recurrent Neural Network)
- **정확도**: 약 99%
- **클래스 수**: 9개
- **특징 추출**: 멜 스펙트로그램 (128 mel bins)
- **샘플레이트**: 22050Hz

## 지원 오디오 형식

- WAV (권장)
- MP3
- FLAC
- OGG
- AIF/AIFF

## 데이터 증강

시간축 이동(Time Shift) 기반 데이터 증강으로 모델 성능을 향상시킬 수 있습니다.

```bash
python scripts/Data\ augmentation\ \(time\ axis\ shift\).py
```

## 문서

- [DRUM_SCORE_ANALYSIS_GUIDE.md](DRUM_SCORE_ANALYSIS_GUIDE.md) - 악보 분석 시스템 상세 가이드
- [ANALYSIS_COMPARISON.md](ANALYSIS_COMPARISON.md) - 세 가지 분석 방법 비교
- [9_class_info.md](9_class_info.md) - 9가지 드럼 클래스 정보

## 활용 예시

- 드럼 커버 연습 (원곡 드럼 패턴 분석)
- 음악 제작 (드럼 패턴 참고)
- 음악 교육 (드럼 악보 자동 생성)
- 음원 분석 연구

## 기술 스택

- **딥러닝**: TensorFlow/Keras
- **오디오 처리**: Librosa
- **데이터 분석**: NumPy, Pandas
- **시각화**: Matplotlib, Seaborn
- **머신러닝**: Scikit-learn

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 감사의 말

이 프로젝트는 다음 오픈소스 라이브러리들을 사용합니다:
- Librosa - 오디오 분석
- TensorFlow/Keras - 딥러닝
- Scikit-learn - 머신러닝
- NumPy, Pandas - 데이터 처리
- Matplotlib, Seaborn - 시각화

---

**Made for Musicians and Audio Engineers**