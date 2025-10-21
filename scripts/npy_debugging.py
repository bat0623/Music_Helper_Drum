import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# ▼▼▼ 1단계: .npy 파일 불러오기 및 기본 정보 확인 ▼▼▼
# =====================================================================

# 파일 경로 설정
DATA_PATH = r"C:\GitHub\Music_Helper_Drum\data_augmented"

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
print("각 클래스별 첫 번째 샘플을 시각화합니다...")
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
            
        # 해당 클래스의 '첫 번째' 샘플 인덱스를 선택
        sample_index = indices_for_label[0]
        
        # 선택된 인덱스의 데이터와 라벨을 가져옴
        sample_feature = X[sample_index]
        sample_label_str = int_to_label[y[sample_index]]

        print(f"인덱스 {sample_index}번 데이터 시각화 (라벨: '{sample_label_str}')")

        # 시각화를 위해 (높이, 너비, 1) -> (높이, 너비) 2차원 배열로 변경
        sample_feature_2d = np.squeeze(sample_feature, axis=-1)

        # matplotlib으로 스펙트로그램 그리기
        plt.figure(figsize=(10, 4))
        plt.imshow(sample_feature_2d, aspect='auto', origin='lower', cmap='viridis')
        
        plt.title(f"Mel Spectrogram (Label: {sample_label_str})")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Bins")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
    
    except Exception as e:
        print(f"'{label_name}' 클래스 시각화 중 오류 발생: {e}")

# 모든 그래프를 화면에 보여줌
plt.show()

print("\n시각화 완료!")