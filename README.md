# Music Helper Drum - 지능형 드럼 트랜스크립션 프로젝트

## 📖 프로젝트 소개

이 프로젝트는 오디오 파일에서 드럼 사운드를 자동으로 검출하고 분류하는 지능형 드럼 트랜스크립션 시스템입니다. 머신러닝 기술을 활용하여 드럼 루프를 분석하고, 각 드럼 타격의 시간과 종류를 정확하게 식별합니다.

### 주요 기능

- 🥁 **자동 드럼 검출**: 오디오에서 드럼 타격 시점 자동 검출
- 🎯 **9개 클래스 드럼 분류**: 
  - **BD** (Bass Drum), **SD** (Snare Drum)
  - **HHC** (Hi-Hat Closed), **HHO** (Hi-Hat Open)
  - **CC** (Crash Cymbal), **RC** (Ride Cymbal)
  - **HT** (High Tom), **MT** (Mid Tom), **LT** (Low Tom)
- 🤖 **두 가지 ML 모델**: Random Forest와 CNN 모델 지원
- 📊 **시각화 도구**: 트랜스크립션 결과 및 분석 시각화
- 🎵 **다양한 출력 형식**: JSON, CSV, MIDI, 텍스트 형식 지원

## 🚀 시작하기

### 필수 요구사항

- Python 3.10.x
- Conda (Anaconda 또는 Miniconda)

### 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/yourusername/Music-Helper-Drum.git
cd Music-Helper-Drum
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

4. **프로젝트 초기 설정**
```bash
python main.py setup
```

## 📁 프로젝트 구조

```
Music Helper Drum/
├── main.py                 # 메인 실행 스크립트
├── requirements.txt        # 패키지 의존성
├── README.md              # 프로젝트 문서
│
├── drum_samples/          # 드럼 샘플 디렉토리 (9개 클래스)
│   ├── bass_drum/         # Bass Drum (BD) 샘플
│   ├── snare_drum/        # Snare Drum (SD) 샘플
│   ├── hihat_closed/      # Hi-Hat Closed (HHC) 샘플
│   ├── hihat_open/        # Hi-Hat Open (HHO) 샘플
│   ├── crash_cymbal/      # Crash Cymbal (CC) 샘플
│   ├── ride_cymbal/       # Ride Cymbal (RC) 샘플
│   ├── high_tom/          # High Tom (HT) 샘플
│   ├── mid_tom/           # Mid Tom (MT) 샘플
│   └── low_tom/           # Low Tom (LT) 샘플
│
├── scripts/               # 주요 스크립트
│   ├── dataset_manager.py    # 데이터셋 관리
│   ├── feature_extractor.py  # 특징 추출
│   ├── train_models.py       # 모델 학습
│   ├── drum_transcription.py # 트랜스크립션
│   └── visualization_tools.py # 시각화 도구
│
├── models/                # 학습된 모델
├── data/                  # 처리된 데이터
└── output/                # 출력 파일
```

## 🎯 사용 방법

### 1. 데이터 준비

드럼 샘플을 각 카테고리별 폴더에 추가 (9개 클래스):
- `drum_samples/bass_drum/` - Bass Drum (BD) 샘플 (.wav, .mp3 등)
- `drum_samples/snare_drum/` - Snare Drum (SD) 샘플
- `drum_samples/hihat_closed/` - Hi-Hat Closed (HHC) 샘플
- `drum_samples/hihat_open/` - Hi-Hat Open (HHO) 샘플
- `drum_samples/crash_cymbal/` - Crash Cymbal (CC) 샘플
- `drum_samples/ride_cymbal/` - Ride Cymbal (RC) 샘플
- `drum_samples/high_tom/` - High Tom (HT) 샘플
- `drum_samples/mid_tom/` - Mid Tom (MT) 샘플
- `drum_samples/low_tom/` - Low Tom (LT) 샘플

### 2. 데이터 처리 및 특징 추출

```bash
python main.py prepare
```

### 3. 모델 학습

```bash
python main.py train
```

### 4. 드럼 트랜스크립션

```bash
# 기본 사용법
python main.py transcribe your_audio.wav

# 시각화 포함
python main.py transcribe your_audio.wav --visualize

# CNN 모델 사용
python main.py transcribe your_audio.wav --model-type keras
```

### 5. 데이터 분석

```bash
python main.py analyze
```

## 🔧 고급 사용법

### 직접 스크립트 실행

개별 스크립트를 직접 실행할 수도 있습니다:

```bash
# 데이터셋 관리
python scripts/dataset_manager.py

# 특징 추출
python scripts/feature_extractor.py

# 모델 학습
python scripts/train_models.py

# 드럼 트랜스크립션
python scripts/drum_transcription.py audio_file.wav
```

### Python에서 모듈로 사용

```python
from scripts.drum_transcription import DrumTranscriber

# 트랜스크라이버 초기화
transcriber = DrumTranscriber(model_type='sklearn', 
                             model_path='models/random_forest_model.pkl')

# 트랜스크립션 수행
transcription = transcriber.transcribe_drum_loop('drum_loop.wav')

# 결과 출력
for event in transcription:
    print(f"시간: {event['time']:.3f}초, 드럼: {event['drum']}, 신뢰도: {event['confidence']:.2f}")
```

## 📊 모델 성능

### Random Forest 모델
- **장점**: 빠른 학습, 적은 데이터로도 좋은 성능
- **특징**: 수작업 특징 추출 (MFCC, 스펙트럼 특징 등)
- **추천**: 실시간 처리가 필요한 경우

### CNN 모델
- **장점**: 높은 정확도, 복잡한 패턴 학습 가능
- **특징**: 멜 스펙트로그램 기반
- **추천**: 최고의 정확도가 필요한 경우

## 🎵 지원 오디오 형식

- WAV (권장)
- MP3
- FLAC
- OGG
- AIF/AIFF

## 📝 출력 형식

- **JSON**: 구조화된 데이터 (프로그래밍 연동용)
- **CSV**: 스프레드시트 분석용
- **MIDI**: DAW 연동용
- **TEXT**: 읽기 쉬운 텍스트 형식

## 🤝 기여하기

프로젝트 개선에 기여를 환영합니다! 
- 버그 리포트
- 기능 제안
- 코드 기여

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 라이브러리들을 사용합니다:
- Librosa - 오디오 분석
- TensorFlow/Keras - 딥러닝
- Scikit-learn - 머신러닝
- NumPy, Pandas - 데이터 처리
- Matplotlib, Seaborn - 시각화

## 📧 문의

프로젝트 관련 문의사항은 이슈 트래커를 이용해주세요.

---

**Made with ❤️ for Musicians and Audio Engineers**
