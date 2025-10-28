import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =====================================================================
# ▼▼▼ 1단계: .npy 파일 불러오기 및 기본 정보 확인 ▼▼▼
# =====================================================================

# 프로젝트 루트 경로 자동 감지
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 파일 경로 설정
DATA_PATH = os.path.join(PROJECT_ROOT, "data_augmented")
DEBUG_IMG_PATH = os.path.join(PROJECT_ROOT, "debug_img")

# 현재 날짜-시간 문자열 생성 (예: 20251026_143025)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 타임스탬프별 폴더 생성 (debug_img/20251026_143025/)
timestamp_folder = os.path.join(DEBUG_IMG_PATH, timestamp)
os.makedirs(timestamp_folder, exist_ok=True)

# 각 클래스별로 저장할 샘플 수
SAMPLES_PER_CLASS = 5

# 라벨-숫자 변환 규칙 정의 (test.py와 동일해야 함)
labels = ['bass_drum', 'crash_cymbal', 'high_tom', 'hihat_closed', 
          'hihat_open', 'low_tom', 'mid_tom', 'ride_cymbal', 'snare_drum']
label_to_int = {label: i for i, label in enumerate(labels)}
int_to_label = {i: label for label, i in label_to_int.items()}

try:
    # .npy 파일 불러오기
    X = np.load(os.path.join(DATA_PATH, "X_data.npy"))
    y = np.load(os.path.join(DATA_PATH, "y_data.npy"))

    print("SUCCESS: .npy 파일 로딩 성공!")
    print(f"X 데이터 형태(Shape): {X.shape}") 
    print(f"y 데이터 형태(Shape): {y.shape}")

except FileNotFoundError:
    print(f"ERROR: '{DATA_PATH}' 폴더에서 .npy 파일을 찾을 수 없습니다.")
    print("먼저 test.py 스크립트를 실행해야 합니다.")
    exit()

# =====================================================================
# ▼▼▼ 2단계: 각 클래스별 첫 번째 샘플 시각화 (수정된 부분) ▼▼▼
# =====================================================================

print("\n" + "="*50)
print(f"각 클래스별로 {SAMPLES_PER_CLASS}개의 샘플을 시각화합니다...")
print("="*50)

# 0번 클래스부터 8번 클래스까지 순회
for label_idx, label_name in enumerate(labels):
    try:
        # 현재 클래스(label_idx)에 해당하는 데이터의 인덱스를 모두 찾음
        # 예: y 배열에서 값이 0인('bass_drum') 요소들의 위치를 모두 찾음
        indices_for_label = np.where(y == label_idx)[0]
        
        # 만약 해당 클래스의 데이터가 하나도 없으면 건너뜀
        if len(indices_for_label) == 0:
            print(f"'{label_name}' 클래스에 해당하는 샘플이 없습니다.")
            continue
        
        # 클래스별 폴더 생성 (debug_img/20251026_143025/bass_drum/)
        class_folder = os.path.join(timestamp_folder, label_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # 해당 클래스의 샘플 개수 확인
        num_samples = min(SAMPLES_PER_CLASS, len(indices_for_label))
        print(f"\n'{label_name}' 클래스: {num_samples}개 샘플 저장 중...")
        
        # 랜덤하게 샘플 인덱스 선택 (중복 없이)
        random_indices = np.random.choice(indices_for_label, size=num_samples, replace=False)
        
        # SAMPLES_PER_CLASS 개수만큼 샘플을 시각화
        for i, sample_index in enumerate(random_indices):
            
            # 선택된 인덱스의 데이터와 라벨을 가져옴
            sample_feature = X[sample_index]
            sample_label_str = int_to_label[y[sample_index]]

            # 시각화를 위해 (높이, 너비, 1) -> (높이, 너비) 2차원 배열로 변경
            sample_feature_2d = np.squeeze(sample_feature, axis=-1)

            # matplotlib으로 스펙트로그램 그리기
            plt.figure(figsize=(10, 4))
            plt.imshow(sample_feature_2d, aspect='auto', origin='lower', cmap='viridis')
            
            plt.title(f"Mel Spectrogram - {sample_label_str} (Sample {i+1}/{num_samples})")
            plt.xlabel("Time Frames")
            plt.ylabel("Mel Bins")
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            
            # 파일명 생성 및 저장 (예: sample_0.png, sample_1.png, ...)
            save_filename = f"sample_{i}.png"
            save_path = os.path.join(class_folder, save_filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()  # 메모리 절약을 위해 figure 닫기
            
            print(f"  [{i+1}/{num_samples}] → {label_name}/sample_{i}.png 저장완료")
    
    except Exception as e:
        print(f"'{label_name}' 클래스 시각화 중 오류 발생: {e}")

print("\n" + "="*50)
print("시각화 완료!")
print(f"저장 위치: {timestamp_folder}")
print(f"각 클래스별로 최대 {SAMPLES_PER_CLASS}개 샘플 저장됨")
print("\n폴더 구조:")
print(f"  debug_img/")
print(f"    └── {timestamp}/")
print(f"        ├── bass_drum/")
print(f"        │   ├── sample_0.png")
print(f"        │   ├── sample_1.png")
print(f"        │   └── ...")
print(f"        ├── crash_cymbal/")
print(f"        └── ... (9개 클래스)")
print("="*50)