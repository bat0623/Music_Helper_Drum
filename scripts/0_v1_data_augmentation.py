import os
import librosa
import numpy as np
from tqdm import tqdm

# =====================================================================
# ▼▼▼ 1. 기본 경로 및 파라미터 설정 ▼▼▼
# =====================================================================
# 프로젝트 루트 경로 자동 감지
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 원본 데이터가 있는 경로
DATA_PATH = os.path.join(PROJECT_ROOT, "drum_samples")
# 증강된 데이터를 저장할 새로운 경로 (v1 버전)
SAVE_PATH = os.path.join(PROJECT_ROOT, "data_augmented_v1")

# 오디오 처리에 사용할 설정값
CONFIG = {
    "sr": 22050,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
}

# =====================================================================
# ▼▼▼ 2. 클래스별 맞춤형 파라미터 설정 (분석 결과 기반) ▼▼▼
# =====================================================================

# 클래스별 최대 길이 설정 (초) - analyze_drum_characteristics.py 분석 결과
CLASS_MAX_DURATION = {
    'bass_drum': 0.509,      # 짧고 강력한 저음
    'snare_drum': 0.557,     # 짧은 타격음 + 스네어 와이어
    'hihat_closed': 0.349,   # 가장 짧은 고음
    'low_tom': 0.714,        # 중간 길이 저음
    'mid_tom': 0.804,        # 중간 길이 중음
    'high_tom': 1.088,       # 중간 길이 고음
    'crash_cymbal': 2.000,   # 긴 잔향 심벌
    'ride_cymbal': 2.000,    # 긴 잔향 심벌
    'hihat_open': 2.000,     # 긴 잔향 하이햇
}

# 클래스별 Trim Threshold (dB) - 에너지 기반
CLASS_TRIM_THRESHOLD = {
    'bass_drum': 35,         # 강한 에너지
    'crash_cymbal': 35,      # 강한 에너지
    'high_tom': 35,          # 강한 에너지
    'hihat_closed': 40,      # 중간 에너지
    'hihat_open': 40,        # 중간 에너지
    'low_tom': 35,           # 강한 에너지
    'mid_tom': 35,           # 강한 에너지
    'ride_cymbal': 40,       # 중간 에너지
    'snare_drum': 40,        # 중간 에너지
}

# =====================================================================
# ▼▼▼ 3. 증강 파라미터 설정 ▼▼▼
# =====================================================================

# 데이터 증강을 적용할 클래스(폴더 이름) 목록 - 모든 클래스 포함
TARGET_CLASSES = ['bass_drum', 'crash_cymbal', 'high_tom', 'hihat_closed', 
                  'hihat_open', 'low_tom', 'mid_tom', 'ride_cymbal', 'snare_drum']

# 위 클래스에 속한 원본 파일 1개당 생성할 증강 데이터 개수
AUGMENTATIONS_PER_FILE = 100

# 증강 기법별 파라미터 설정
AUGMENTATION_PARAMS = {
    "time_shift_seconds": 0.2,      # 시간 축 이동 범위 (초)
    "noise_factor": 0.005,           # 잡음 강도
    "pitch_shift_steps": 2,          # 음높이 변경 범위 (-2 ~ +2 반음)
    "time_stretch_rate": 0.1,        # 속도 변경 범위 (0.9 ~ 1.1배)
    "volume_change_db": 3            # 볼륨 변경 범위 (-3 ~ +3 dB)
}

# =====================================================================
# ▼▼▼ 4. 증강 함수 정의 ▼▼▼
# =====================================================================

def time_shift(audio, sr, shift_max_seconds):
    """오디오 데이터의 시간 축을 랜덤하게 이동시킵니다."""
    shift_limit = int(sr * shift_max_seconds)
    shift_amount = int(np.random.rand() * shift_limit)
    shifted_audio = np.roll(audio, shift_amount)
    return shifted_audio

def add_noise(audio, noise_factor):
    """오디오에 랜덤 노이즈를 추가합니다."""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    # 클리핑 방지 (-1.0 ~ 1.0 범위)
    augmented_audio = np.clip(augmented_audio, -1.0, 1.0)
    return augmented_audio

def pitch_shift(audio, sr, n_steps):
    """음높이를 반음 단위로 변경합니다."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate):
    """오디오의 속도를 변경합니다 (음높이는 유지)."""
    return librosa.effects.time_stretch(audio, rate=rate)

def change_volume(audio, db_change):
    """볼륨을 dB 단위로 변경합니다."""
    return audio * (10 ** (db_change / 20.0))

# =====================================================================
# ▼▼▼ 5. 클래스별 맞춤형 전처리 함수 ▼▼▼
# =====================================================================

def preprocess_audio(audio, sr, label):
    """
    클래스별 맞춤형 전처리 적용
    1. Energy-based Trimming (클래스별 threshold)
    2. 최대 길이 제한 (클래스별)
    """
    # 1. 클래스별 threshold로 trim
    trim_threshold = CLASS_TRIM_THRESHOLD.get(label, 40)
    y_trimmed, _ = librosa.effects.trim(audio, top_db=trim_threshold)
    
    # 2. 최대 길이 제한
    max_duration = CLASS_MAX_DURATION.get(label, 2.0)
    max_samples = int(sr * max_duration)
    
    if len(y_trimmed) > max_samples:
        y_trimmed = y_trimmed[:max_samples]
    
    return y_trimmed

def audio_to_melspec(audio_data, sr):
    """
    오디오를 멜 스펙트로그램으로 변환 (동적 길이)
    """
    melspec = librosa.feature.melspectrogram(
        y=audio_data, 
        sr=sr, 
        n_mels=CONFIG["n_mels"], 
        n_fft=CONFIG["n_fft"], 
        hop_length=CONFIG["hop_length"]
    )
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    return melspec_db

def pad_melspec(melspec_db, target_frames):
    """
    멜 스펙트로그램을 목표 프레임 수에 맞춰 패딩 또는 자르기
    """
    if melspec_db.shape[1] > target_frames:
        melspec_db = melspec_db[:, :target_frames]
    else:
        pad_width = target_frames - melspec_db.shape[1]
        melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return melspec_db

# =====================================================================
# ▼▼▼ 6. 데이터 전처리 및 증강 실행 ▼▼▼
# =====================================================================

print("="*70)
print("V1: 클래스별 맞춤형 데이터 증강 시작")
print("="*70)
print("\n[개선사항]")
print("  1. 클래스별 Energy-based Trimming 적용")
print("  2. 클래스별 최대 길이 제한")
print("  3. 불필요한 무음 구간 제거")
print("  4. 5가지 증강 기법 랜덤 조합")
print("="*70)

# 전처리 및 증강된 데이터를 저장할 리스트 초기화
all_features = []
all_labels = []

# 통계 정보 수집
trim_stats = {}

# 라벨(폴더) 정보 읽기 및 숫자 매핑
labels = [label for label in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, label))]
label_to_int = {label: i for i, label in enumerate(labels)}
print("\n라벨 매핑:", label_to_int)

# 1단계: 최대 프레임 길이 계산 (전체 데이터 스캔)
print("\n[1단계] 최대 프레임 길이 계산 중...")
max_frames = 0

for label in labels:
    label_path = os.path.join(DATA_PATH, label)
    wav_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]
    
    original_lengths = []
    trimmed_lengths = []
    
    for wav_file in wav_files[:10]:  # 샘플링 (전체 스캔은 느리므로)
        try:
            file_path = os.path.join(label_path, wav_file)
            y, sr = librosa.load(file_path, sr=CONFIG["sr"])
            
            original_lengths.append(len(y))
            
            # 전처리 적용
            y_preprocessed = preprocess_audio(y, sr, label)
            trimmed_lengths.append(len(y_preprocessed))
            
            # 멜 스펙트로그램 변환
            melspec = audio_to_melspec(y_preprocessed, sr)
            max_frames = max(max_frames, melspec.shape[1])
            
        except Exception as e:
            continue
    
    if len(trimmed_lengths) > 0:
        avg_reduction = (1 - np.mean(trimmed_lengths) / np.mean(original_lengths)) * 100
        trim_stats[label] = {
            'avg_reduction': avg_reduction,
            'original': np.mean(original_lengths),
            'trimmed': np.mean(trimmed_lengths)
        }

print(f"\n계산된 최대 프레임 수: {max_frames}")
print("\n클래스별 평균 데이터 감소율:")
for label, stats in trim_stats.items():
    print(f"  {label:<15}: {stats['avg_reduction']:.1f}% 감소 "
          f"({stats['original']:.0f} -> {stats['trimmed']:.0f} samples)")

# 2단계: 실제 데이터 전처리 및 증강
print(f"\n[2단계] 데이터 전처리 및 증강 실행 (목표 프레임: {max_frames})")

for label in labels:
    label_path = os.path.join(DATA_PATH, label)
    wav_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]

    is_target_class = label in TARGET_CLASSES
    if is_target_class:
        print(f"\n'{label}' 클래스 증강 진행... (원본 1개당 {AUGMENTATIONS_PER_FILE}개 생성)")
        print(f"  - Trim Threshold: {CLASS_TRIM_THRESHOLD.get(label, 40)}dB")
        print(f"  - 최대 길이: {CLASS_MAX_DURATION.get(label, 2.0):.3f}초")
    else:
        print(f"\n'{label}' 클래스 처리 중...")

    for wav_file in tqdm(wav_files, desc=f"Processing {label}"):
        try:
            file_path = os.path.join(label_path, wav_file)
            y, sr = librosa.load(file_path, sr=CONFIG["sr"])
            
            # 클래스별 맞춤형 전처리 적용
            y_preprocessed = preprocess_audio(y, sr, label)
            
            # 원본 데이터 처리
            melspec_original = audio_to_melspec(y_preprocessed, sr)
            melspec_padded = pad_melspec(melspec_original, max_frames)
            all_features.append(melspec_padded)
            all_labels.append(label_to_int[label])

            # 타겟 클래스인 경우에만 증강 실행
            if is_target_class:
                for _ in range(AUGMENTATIONS_PER_FILE):
                    augmented_y = y_preprocessed.copy()
                    
                    # 랜덤하게 1~3개의 증강 기법을 조합하여 적용
                    num_augmentations = np.random.randint(1, 4)  # 1, 2, 또는 3개 선택
                    augmentation_choices = np.random.choice(5, size=num_augmentations, replace=False)
                    
                    for aug_choice in augmentation_choices:
                        if aug_choice == 0:  # Time Shift
                            augmented_y = time_shift(augmented_y, sr, AUGMENTATION_PARAMS["time_shift_seconds"])
                        elif aug_choice == 1:  # Add Noise
                            augmented_y = add_noise(augmented_y, AUGMENTATION_PARAMS["noise_factor"])
                        elif aug_choice == 2:  # Pitch Shift
                            n_steps = np.random.uniform(-AUGMENTATION_PARAMS["pitch_shift_steps"], 
                                                       AUGMENTATION_PARAMS["pitch_shift_steps"])
                            augmented_y = pitch_shift(augmented_y, sr, n_steps)
                        elif aug_choice == 3:  # Time Stretch
                            rate = np.random.uniform(1.0 - AUGMENTATION_PARAMS["time_stretch_rate"], 
                                                    1.0 + AUGMENTATION_PARAMS["time_stretch_rate"])
                            augmented_y = time_stretch(augmented_y, rate)
                        elif aug_choice == 4:  # Volume Change
                            db_change = np.random.uniform(-AUGMENTATION_PARAMS["volume_change_db"], 
                                                         AUGMENTATION_PARAMS["volume_change_db"])
                            augmented_y = change_volume(augmented_y, db_change)
                    
                    # 증강된 오디오를 멜 스펙트로그램으로 변환
                    melspec_augmented = audio_to_melspec(augmented_y, sr)
                    melspec_padded = pad_melspec(melspec_augmented, max_frames)
                    all_features.append(melspec_padded)
                    all_labels.append(label_to_int[label])

        except Exception as e:
            print(f"  [ERROR] {wav_file} 처리 중 오류 발생: {e}")

# =====================================================================
# ▼▼▼ 7. 최종 데이터 변환 및 저장 ▼▼▼
# =====================================================================
X = np.array(all_features)
y = np.array(all_labels)
X = X[..., np.newaxis]  # CNN 모델을 위한 채널 차원 추가

print("\n" + "="*70)
print("증강 및 전처리 완료!")
print("="*70)
print(f"총 데이터 개수: {len(X)}개")
print(f"특성 데이터(X) Shape: {X.shape}")
print(f"라벨 데이터(y) Shape: {y.shape}")
print(f"실제 프레임 수: {X.shape[2]} (기존 200에서 최적화)")

# 증강 후 클래스별 샘플 수 확인
print(f"\n증강 후 드럼 클래스별 샘플 수:")
for label, idx in label_to_int.items():
    count = np.sum(y == idx)
    print(f"  {label:<15}: {count:>6}개 샘플")

# 증강된 데이터 파일로 저장
os.makedirs(SAVE_PATH, exist_ok=True)
np.save(os.path.join(SAVE_PATH, "X_data.npy"), X)
np.save(os.path.join(SAVE_PATH, "y_data.npy"), y)

print("\n" + "="*70)
print(f"'{SAVE_PATH}' 폴더에 v1 증강 데이터가 저장되었습니다.")
print("파일명: X_data.npy, y_data.npy")
print("="*70)

# V0 vs V1 비교 정보 출력
print("\n[V0 vs V1 비교]")
print(f"  V0 (기존): 고정 길이 200 프레임")
print(f"  V1 (개선): 최적화된 {X.shape[2]} 프레임")
print(f"  예상 메모리 절약: {(1 - X.shape[2]/200) * 100:.1f}%")
print(f"  예상 학습 속도 향상: {200/X.shape[2]:.1f}배")
print("="*70)

