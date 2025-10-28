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
# 증강된 데이터를 저장할 새로운 경로
SAVE_PATH = os.path.join(PROJECT_ROOT, "data_augmented")

# 오디오 처리에 사용할 설정값
CONFIG = {
    "sr": 22050,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "max_len": 200
}

# =====================================================================
# ▼▼▼ 2. 증강 파라미터 설정 (사용자 설정 구간) ▼▼▼
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
# ▼▼▼ 3. 증강 함수 정의 ▼▼▼
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
# ▼▼▼ 4. 데이터 전처리 및 증강 실행 ▼▼▼
# =====================================================================
# 전처리 및 증강된 데이터를 저장할 리스트 초기화
all_features = []
all_labels = []

# 라벨(폴더) 정보 읽기 및 숫자 매핑
labels = [label for label in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, label))]
label_to_int = {label: i for i, label in enumerate(labels)}
print("라벨 매핑:", label_to_int)

# 각 폴더(라벨)별로 음성 파일 처리
for label in labels:
    label_path = os.path.join(DATA_PATH, label)
    wav_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]

    is_target_class = label in TARGET_CLASSES
    if is_target_class:
        print(f"\n'{label}' 클래스 증강 진행... (원본 1개당 {AUGMENTATIONS_PER_FILE}개 생성)")
    else:
        print(f"\n'{label}' 클래스 처리 중...")

    for wav_file in tqdm(wav_files, desc=f"Processing {label}"):
        try:
            file_path = os.path.join(label_path, wav_file)
            y, sr = librosa.load(file_path, sr=CONFIG["sr"])

            # 원본 데이터는 항상 포함
            # 멜 스펙트로그램 변환 및 크기 조절 함수
            def process_audio(audio_data):
                melspec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=CONFIG["n_mels"], n_fft=CONFIG["n_fft"], hop_length=CONFIG["hop_length"])
                melspec_db = librosa.power_to_db(melspec, ref=np.max)

                if melspec_db.shape[1] > CONFIG["max_len"]:
                    melspec_db = melspec_db[:, :CONFIG["max_len"]]
                else:
                    pad_width = CONFIG["max_len"] - melspec_db.shape[1]
                    melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
                return melspec_db

            # 원본 데이터 처리
            original_feature = process_audio(y)
            all_features.append(original_feature)
            all_labels.append(label_to_int[label])

            # 타겟 클래스인 경우에만 증강 실행
            if is_target_class:
                for _ in range(AUGMENTATIONS_PER_FILE):
                    augmented_y = y.copy()
                    
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
                    
                    augmented_feature = process_audio(augmented_y)
                    all_features.append(augmented_feature)
                    all_labels.append(label_to_int[label])

        except Exception as e:
            print(f"{wav_file} 처리 중 오류 발생: {e}")

# =====================================================================
# ▼▼▼ 5. 최종 데이터 변환 및 저장 ▼▼▼
# =====================================================================
X = np.array(all_features)
y = np.array(all_labels)
X = X[..., np.newaxis] # CNN 모델을 위한 채널 차원 추가

print("\n" + "="*50)
print("증강 및 전처리 완료!")
print(f"총 데이터 개수: {len(X)}개")
print("특성 데이터(X) Shape:", X.shape)
print("라벨 데이터(y) Shape:", y.shape)

# 증강 후 클래스별 샘플 수 확인
print(f"\n증강 후 드럼 클래스별 샘플 수:")
for label, idx in label_to_int.items():
    count = np.sum(y == idx)
    print(f"  {label}: {count}개 샘플")

# 증강된 데이터 파일로 저장
os.makedirs(SAVE_PATH, exist_ok=True)
np.save(os.path.join(SAVE_PATH, "X_data.npy"), X)
np.save(os.path.join(SAVE_PATH, "y_data.npy"), y)

print(f"\n'{SAVE_PATH}' 폴더에 증강된 데이터가 저장되었습니다.")
print("파일명: X_data.npy, y_data.npy")