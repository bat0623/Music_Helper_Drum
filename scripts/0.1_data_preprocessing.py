import os
import librosa
import numpy as np
from tqdm import tqdm

# =====================================================================
# ▼▼▼ 1. 파일 경로 설정 ▼▼▼
# =====================================================================
# 프로젝트 루트 경로 자동 감지
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 원본 데이터가 있는 경로
DATA_PATH = os.path.join(PROJECT_ROOT, "drum_samples")
# 전처리된 데이터를 저장할 경로
SAVE_PATH = os.path.join(PROJECT_ROOT, "data")

# 전처리에 사용할 파라미터 정의
CONFIG = {
    "sr": 22050,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "max_len": 200
}

# 전처리된 데이터를 저장할 리스트 초기화
all_features = []
all_labels = []

# =====================================================================
# ▼▼▼ 2. 운영체제(OS)에 맞는 경로 결합 ▼▼▼
# =====================================================================
# os.path.join() 함수는 현재 실행 중인 운영체제(Windows, Mac, Linux)를
# 자동으로 파악해서 그에 맞는 경로 구분자('\', '/')를 사용해 경로를 합쳐줍니다.
# 윈도우에서는 DATA_PATH와 하위 폴더 이름 사이에 '\'를 넣어줍니다. (예: "C:\...\bass_drum")
# 따라서 코드를 수정 없이 다른 OS에서도 사용할 수 있게 해주는 매우 중요한 부분입니다.
labels = [label for label in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, label))]
label_to_int = {label: i for i, label in enumerate(labels)}
print("라벨 매핑:", label_to_int)

# 각 폴더(라벨)별로 음성 파일 처리
for label in labels:
    label_path = os.path.join(DATA_PATH, label)
    wav_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]

    print(f"\n'{label}' 폴더 처리 중...")
    for wav_file in tqdm(wav_files):
        try:
            # 여기서도 os.path.join을 사용해 완전한 파일 경로를 만듭니다.
            file_path = os.path.join(label_path, wav_file)

            # 1. 음성 파일 불러오기
            y, sr = librosa.load(file_path, sr=CONFIG["sr"])

            # 2. 멜 스펙트로그램 추출
            melspec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=CONFIG["n_mels"],
                n_fft=CONFIG["n_fft"],
                hop_length=CONFIG["hop_length"]
            )
            melspec_db = librosa.power_to_db(melspec, ref=np.max)

            # 3. 길이 통일
            if melspec_db.shape[1] > CONFIG["max_len"]:
                melspec_db = melspec_db[:, :CONFIG["max_len"]]
            else:
                pad_width = CONFIG["max_len"] - melspec_db.shape[1]
                melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

            # 4. 데이터와 라벨 저장
            all_features.append(melspec_db)
            all_labels.append(label_to_int[label])

        except Exception as e:
            print(f"{wav_file} 처리 중 오류 발생: {e}")

# 리스트를 Numpy 배열로 변환
X = np.array(all_features)
y = np.array(all_labels)

# 채널 차원 추가
X = X[..., np.newaxis]

print("\n전처리 완료!")
print("특성 데이터(X) Shape:", X.shape)
print("라벨 데이터(y) Shape:", y.shape)

# 5. 전처리된 데이터 저장
# data 폴더가 없으면 생성
os.makedirs(SAVE_PATH, exist_ok=True)

# data 폴더에 .npy 파일들을 저장합니다.
np.save(os.path.join(SAVE_PATH, "X_data.npy"), X)
np.save(os.path.join(SAVE_PATH, "y_data.npy"), y)

print(f"\n'{SAVE_PATH}' 폴더에 'X_data.npy'와 'y_data.npy' 파일로 데이터가 저장되었습니다.")