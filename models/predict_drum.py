import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras

# =====================================================================
# ▼▼▼ 드럼 사운드 예측 클래스 ▼▼▼
# =====================================================================

class DrumPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.labels = ['bass_drum', 'crash_cymbal', 'high_tom', 'hihat_closed', 
                      'hihat_open', 'low_tom', 'mid_tom', 'ride_cymbal', 'snare_drum']
        
        # 전처리 설정 (훈련 시와 동일해야 함)
        self.config = {
            "sr": 22050,
            "n_mels": 128,
            "n_fft": 2048,
            "hop_length": 512,
            "max_len": 200
        }
        
        self.load_model()
    
    def load_model(self):
        """훈련된 모델 로딩"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"모델 로딩 성공: {self.model_path}")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise
    
    def preprocess_audio(self, audio_path):
        """오디오 파일 전처리"""
        try:
            # 오디오 파일 로딩
            y, sr = librosa.load(audio_path, sr=self.config["sr"])
            
            # 멜 스펙트로그램 변환
            melspec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.config["n_mels"],
                n_fft=self.config["n_fft"],
                hop_length=self.config["hop_length"]
            )
            melspec_db = librosa.power_to_db(melspec, ref=np.max)
            
            # 길이 통일
            if melspec_db.shape[1] > self.config["max_len"]:
                melspec_db = melspec_db[:, :self.config["max_len"]]
            else:
                pad_width = self.config["max_len"] - melspec_db.shape[1]
                melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
            # 배치 차원과 채널 차원 추가
            melspec_db = melspec_db[np.newaxis, ..., np.newaxis]
            
            return melspec_db
            
        except Exception as e:
            print(f"오디오 전처리 실패: {e}")
            return None
    
    def predict(self, audio_path, top_k=3):
        """단일 오디오 파일 예측"""
        # 전처리
        processed_audio = self.preprocess_audio(audio_path)
        if processed_audio is None:
            return None
        
        # 예측 수행
        predictions = self.model.predict(processed_audio, verbose=0)
        probabilities = predictions[0]
        
        # 상위 k개 결과 정렬
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'class': self.labels[idx],
                'probability': float(probabilities[idx]),
                'confidence': f"{probabilities[idx]*100:.2f}%"
            })
        
        return results
    
    def predict_batch(self, audio_paths):
        """여러 오디오 파일 일괄 예측"""
        all_results = {}
        
        for audio_path in audio_paths:
            print(f"예측 중: {os.path.basename(audio_path)}")
            results = self.predict(audio_path)
            all_results[audio_path] = results
        
        return all_results

# =====================================================================
# ▼▼▼ 사용 예시 함수 ▼▼▼
# =====================================================================

def demo_prediction():
    """예측 데모 함수"""
    # 모델 경로 설정
    MODEL_PATH = r"C:\GitHub\Music Helper Drum\models\final_crnn_drum_model.h5"
    
    # 예측기 초기화
    try:
        predictor = DrumPredictor(MODEL_PATH)
    except:
        print("모델 파일이 없습니다. 먼저 crnn_drum_classifier.py를 실행해서 모델을 훈련하세요.")
        return
    
    # 테스트할 오디오 파일 경로들
    test_audio_paths = [
        r"C:\GitHub\Music Helper Drum\drum_samples\bass_drum\bass drum.wav",
        r"C:\GitHub\Music Helper Drum\drum_samples\snare_drum\ASR-X Snare 02.wav",
        r"C:\GitHub\Music Helper Drum\drum_samples\hihat_closed\Closed Hihat1.wav"
    ]
    
    print("="*60)
    print("드럼 사운드 예측 데모")
    print("="*60)
    
    for audio_path in test_audio_paths:
        if os.path.exists(audio_path):
            print(f"\n파일: {os.path.basename(audio_path)}")
            print("-" * 40)
            
            results = predictor.predict(audio_path, top_k=3)
            
            if results:
                for result in results:
                    print(f"{result['rank']}위: {result['class']} ({result['confidence']})")
            else:
                print("예측 실패")
        else:
            print(f"파일을 찾을 수 없습니다: {audio_path}")

def interactive_prediction():
    """대화형 예측 함수"""
    MODEL_PATH = r"C:\GitHub\Music Helper Drum\models\final_crnn_drum_model.h5"
    
    try:
        predictor = DrumPredictor(MODEL_PATH)
    except:
        print("모델 파일이 없습니다. 먼저 crnn_drum_classifier.py를 실행해서 모델을 훈련하세요.")
        return
    
    print("="*60)
    print("대화형 드럼 사운드 예측")
    print("종료하려면 'quit' 입력")
    print("="*60)
    
    while True:
        audio_path = input("\n오디오 파일 경로를 입력하세요: ").strip()
        
        if audio_path.lower() == 'quit':
            print("예측을 종료합니다.")
            break
        
        if not os.path.exists(audio_path):
            print("파일을 찾을 수 없습니다.")
            continue
        
        print(f"\n예측 중: {os.path.basename(audio_path)}")
        results = predictor.predict(audio_path, top_k=5)
        
        if results:
            print("\n예측 결과:")
            for result in results:
                print(f"  {result['rank']}위: {result['class']} ({result['confidence']})")
        else:
            print("예측 실패")

if __name__ == "__main__":
    # 데모 실행
    demo_prediction()
    
    # 대화형 예측 (선택사항)
    # interactive_prediction()
