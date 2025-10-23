# Music Helper Drum - 빠른 시작 가이드

## 파일명 체계

모든 파일이 실행 순서대로 번호가 매겨져 있습니다.

### scripts/ 폴더
```
0_data_augmentation.py         → 데이터 증강
0.1_data_preprocessing.py      → 기본 데이터 전처리
0.2_data_debugging.py          → 데이터 검증 및 시각화
3.1_test_continuous.py         → 연속 음원 분석 테스트
4.1_test_score.py              → 악보 분석 테스트
```

### models/ 폴더
```
1_model_training.py            → 모델 학습
2_model_predict.py             → 단일 드럼 예측
3_continuous_analyzer.py       → 연속 음원 분석기
4_score_analyzer.py            → 악보 분석기
```

---

## 실행 순서

### 처음 사용하는 경우:

```bash
# 1단계: 데이터 증강
python scripts/0_data_augmentation.py

# 2단계: 모델 학습
python models/1_model_training.py

# 3단계: 단일 드럼 예측 테스트
python models/2_model_predict.py

# 4단계: 연속 음원 분석
python models/3_continuous_analyzer.py

# 5단계: 악보 분석
python models/4_score_analyzer.py
```

### 모델이 이미 학습된 경우:

```bash
# 예측만 실행
python models/2_model_predict.py

# 또는 연속 음원 분석
python models/3_continuous_analyzer.py

# 또는 악보 분석
python models/4_score_analyzer.py
```

---

## 각 단계별 설명

### 0단계: 데이터 증강
- **목적**: 학습 데이터 양을 늘려 모델 성능 향상
- **실행 시간**: 수 분
- **결과**: `data_augmented/` 폴더에 `.npy` 파일 생성

### 1단계: 모델 학습
- **목적**: 드럼 분류 모델 학습
- **실행 시간**: 30분~1시간 (GPU 사용 시 더 빠름)
- **결과**: `models/` 폴더에 `.h5` 모델 파일 생성

### 2단계: 단일 예측
- **목적**: 짧은 드럼 사운드 1개씩 분류
- **입력**: `TestSound/` 폴더의 오디오 파일
- **결과**: 콘솔에 예측 결과 출력

### 3단계: 연속 음원 분석
- **목적**: 긴 드럼 연주 음원을 시간대별로 분석
- **입력**: `TestSound/` 폴더의 오디오 파일
- **결과**: `output/` 폴더에 PNG 이미지와 TXT 파일 생성

### 4단계: 악보 분석
- **목적**: 연주 음원을 악보 형식으로 상세 분석
- **입력**: `TestSound/` 폴더의 오디오 파일
- **결과**: `output/` 폴더에 PNG 이미지와 TXT 파일 생성

---

## 중요 폴더

| 폴더 | 설명 |
|------|------|
| `drum_samples/` | 원본 드럼 샘플 (9개 클래스) |
| `data_augmented/` | 증강된 학습 데이터 |
| `TestSound/` | 테스트할 오디오 파일 넣는 곳 |
| `output/` | 분석 결과가 저장되는 곳 |
| `models/` | 학습된 모델과 코드 |
| `scripts/` | 데이터 처리 스크립트 |

---

## 9개 드럼 클래스

1. **BD** - 베이스드럼 (Bass Drum)
2. **SD** - 스네어드럼 (Snare Drum)
3. **HT** - 하이톰 (High Tom)
4. **MT** - 미드톰 (Mid Tom)
5. **LT** - 로우톰 (Low Tom)
6. **HH** - 클로즈드하이햇 (Hihat Closed)
7. **OH** - 오픈하이햇 (Hihat Open)
8. **CC** - 크래시심벌 (Crash Cymbal)
9. **RC** - 라이드심벌 (Ride Cymbal)

---

## 자주 사용하는 명령어

```bash
# 새로운 오디오 파일 분석
python models/4_score_analyzer.py

# 모델 재학습 (데이터 추가 후)
python scripts/0_data_augmentation.py
python models/1_model_training.py

# 데이터 검증
python scripts/0.2_data_debugging.py
```

---

## 팁

1. **테스트 파일 추가**: `TestSound/` 폴더에 `.wav` 파일을 넣으면 자동으로 분석됨
2. **결과 확인**: `output/` 폴더에서 이미지와 텍스트 파일로 확인
3. **모델 개선**: 더 많은 샘플을 `drum_samples/`에 추가하고 재학습
4. **빠른 테스트**: 2단계부터 시작 (모델이 이미 있으면)

---

## 문제 해결

**"모델 파일을 찾을 수 없습니다"**
→ `python models/1_model_training.py` 먼저 실행

**"데이터 파일을 찾을 수 없습니다"**
→ `python scripts/0_data_augmentation.py` 먼저 실행

**결과가 부정확함**
→ 더 많은 학습 데이터 추가 또는 모델 재학습

---

상세한 내용은 `PROJECT_WORKFLOW.md` 파일을 참고하세요.
