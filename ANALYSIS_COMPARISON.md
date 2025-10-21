# 드럼 분석 시스템 비교

## 세 가지 분석 방법

프로젝트에는 세 가지 드럼 분석 시스템이 있습니다. 목적에 따라 적합한 시스템을 선택하세요.

## 1. predict_drum.py - 단일 드럼 사운드 분류

### 용도
개별 드럼 사운드 파일을 분류

### 특징
- 하나의 드럼 사운드만 포함된 파일 분석
- 빠른 예측 속도
- 상위 3개 예측 결과 제공

### 사용 예시
```python
from models.predict_drum import DrumPredictor

predictor = DrumPredictor("models/final_crnn_drum_model.h5")
results = predictor.predict("drum_samples/snare_drum/snare1.wav")
```

### 적합한 경우
- 단일 드럼 샘플 분류
- 드럼 샘플 라이브러리 정리
- 빠른 테스트

---

## 2. continuous_drum_analyzer.py - 연속 음원 분석

### 용도
연속된 음원에서 드럼 사운드 감지 및 분류

### 특징
- 슬라이딩 윈도우 방식
- 온셋 기반 방식
- 두 가지 방식 비교 가능

### 분석 방식

#### A. 슬라이딩 윈도우
- 일정 간격(기본 2초)으로 음원 분할
- 0.5초씩 이동하며 분석
- 모든 구간을 균등하게 분석

#### B. 온셋 기반
- 드럼 사운드 시작점 자동 감지
- 해당 구간만 분석
- 계산량 적음

### 사용 예시
```python
from models.continuous_drum_analyzer import ContinuousDrumAnalyzer

analyzer = ContinuousDrumAnalyzer("models/final_crnn_drum_model.h5")
results, duration = analyzer.analyze_continuous_audio("audio.wav", method='sliding_window')
```

### 적합한 경우
- 일반적인 연속 음원 분석
- 두 가지 방식 비교 필요
- 연구 및 실험 목적

---

## 3. drum_score_analyzer.py - 드럼 악보 분석 (권장)

### 용도
연속 음원을 악보처럼 분석하고 표시

### 특징
- 정밀한 분석 (0.5초 윈도우, 0.05초 간격)
- 모든 드럼 사운드 빠짐없이 감지
- 악보 스타일 타임라인 출력
- 0.1초 단위 상세 타임라인
- 자동 중복 제거

### 출력 형식
```
[타임라인 - 악보 형식]
시간(초)   | 드럼 종류            | 신뢰도    
  0.00     | CC (크래시심벌)      | 91.5%     
  0.16     | MT (미드톰)          | 100.0%    
  0.42     | LT (로우톰)          | 100.0%    

[상세 타임라인 - 0.1초 단위]
 0.00s | CC
 0.10s | MT
 0.20s | MT
 0.40s | LT
```

### 사용 예시
```python
from models.drum_score_analyzer import DrumScoreAnalyzer

analyzer = DrumScoreAnalyzer("models/final_crnn_drum_model.h5")
results, duration = analyzer.analyze_continuous_audio("audio.wav")
analyzer.print_score_timeline(results, duration)
```

### 적합한 경우
- 정밀한 드럼 패턴 분석 (권장)
- 악보 형식 출력 필요
- 드럼 커버 연습
- 음악 제작 참고

---

## 상세 비교표

| 항목 | predict_drum | continuous_drum | drum_score |
|------|-------------|-----------------|------------|
| **분석 대상** | 단일 드럼 | 연속 음원 | 연속 음원 |
| **윈도우 크기** | 전체 | 2.0초 | 0.5초 |
| **이동 간격** | - | 0.5초 | 0.05초 |
| **분석 정밀도** | 높음 | 중간 | 매우 높음 |
| **분석 속도** | 빠름 | 중간 | 느림 |
| **출력 형식** | 예측 확률 | 시간대별 감지 | 악보 스타일 |
| **중복 제거** | - | 수동 | 자동 |
| **상세 타임라인** | - | X | O (0.1초 단위) |
| **시각화** | X | O | O (3가지) |
| **클래스별 통계** | X | O | O |
| **파일 출력** | X | O | O |

---

## 선택 가이드

### 단일 드럼 샘플 분류
→ **predict_drum.py** 사용

### 연속 음원에서 드럼 감지 (실험)
→ **continuous_drum_analyzer.py** 사용
- 슬라이딩 윈도우 vs 온셋 기반 비교
- 연구 목적

### 연속 음원 정밀 분석 (실용)
→ **drum_score_analyzer.py** 사용 (권장)
- 모든 드럼 사운드 빠짐없이 감지
- 악보처럼 보기 쉬운 출력
- 드럼 커버, 음악 제작에 최적

---

## 성능 비교 (2초 음원 기준)

| 시스템 | 분석 시간 | 감지 개수 | 정확도 |
|--------|-----------|-----------|--------|
| continuous (sliding) | ~5초 | 1-3개 | 중간 |
| continuous (onset) | ~3초 | 1-5개 | 중간 |
| drum_score | ~15초 | 5-20개 | 높음 |

---

## 파라미터 비교

### continuous_drum_analyzer.py
```python
window_size = 2.0           # 윈도우 크기
hop_size = 0.5              # 이동 간격
confidence_threshold = 0.7  # 신뢰도 임계값
```

### drum_score_analyzer.py
```python
window_size = 0.5           # 윈도우 크기 (더 작음)
hop_size = 0.05             # 이동 간격 (더 조밀)
confidence_threshold = 0.5  # 신뢰도 임계값 (더 낮음)
```

더 작은 윈도우와 조밀한 간격으로 **더 많은 드럼 사운드를 감지**합니다.

---

## 결론

- **단일 드럼 분류**: predict_drum.py
- **실험 및 연구**: continuous_drum_analyzer.py
- **실용적 분석**: drum_score_analyzer.py (권장)

대부분의 경우 **drum_score_analyzer.py**를 사용하는 것을 권장합니다.


