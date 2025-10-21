import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# ▼▼▼ 연속 음원 드럼 분석기 클래스 ▼▼▼
# =====================================================================

class ContinuousDrumAnalyzer:
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
        
        # 분석 설정
        self.window_size = 2.0  # 윈도우 크기 (초)
        self.hop_size = 0.5     # 윈도우 이동 간격 (초)
        self.confidence_threshold = 0.7  # 신뢰도 임계값
        
        self.load_model()
    
    def load_model(self):
        """훈련된 모델 로딩"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"모델 로딩 성공: {self.model_path}")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise
    
    def detect_onsets(self, audio, sr):
        """오디오에서 드럼 사운드의 시작점(온셋) 감지"""
        # RMS 에너지 계산
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 온셋 감지
        onsets = librosa.onset.onset_detect(
            y=audio, 
            sr=sr, 
            hop_length=hop_length,
            units='time',
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.2,
            wait=10
        )
        
        return onsets, rms
    
    def preprocess_segment(self, audio_segment, sr):
        """오디오 세그먼트 전처리"""
        try:
            # 멜 스펙트로그램 변환
            melspec = librosa.feature.melspectrogram(
                y=audio_segment,
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
            print(f"세그먼트 전처리 실패: {e}")
            return None
    
    def analyze_continuous_audio(self, audio_path, method='sliding_window'):
        """연속 음원 분석"""
        print(f"연속 음원 분석 시작: {os.path.basename(audio_path)}")
        
        # 오디오 로딩
        audio, sr = librosa.load(audio_path, sr=self.config["sr"])
        duration = len(audio) / sr
        
        print(f"오디오 길이: {duration:.2f}초")
        
        results = []
        
        if method == 'sliding_window':
            results = self._sliding_window_analysis(audio, sr)
        elif method == 'onset_based':
            results = self._onset_based_analysis(audio, sr)
        else:
            raise ValueError("method는 'sliding_window' 또는 'onset_based'여야 합니다.")
        
        return results, duration
    
    def _sliding_window_analysis(self, audio, sr):
        """슬라이딩 윈도우 방식 분석"""
        print("슬라이딩 윈도우 방식으로 분석 중...")
        
        window_samples = int(self.window_size * sr)
        hop_samples = int(self.hop_size * sr)
        
        results = []
        total_windows = (len(audio) - window_samples) // hop_samples + 1
        
        for i in range(0, len(audio) - window_samples + 1, hop_samples):
            start_time = i / sr
            end_time = (i + window_samples) / sr
            
            # 윈도우 추출
            window = audio[i:i + window_samples]
            
            # 전처리
            processed = self.preprocess_segment(window, sr)
            if processed is None:
                continue
            
            # 예측
            predictions = self.model.predict(processed, verbose=0)
            probabilities = predictions[0]
            
            # 최고 신뢰도 클래스 찾기
            max_idx = np.argmax(probabilities)
            max_prob = probabilities[max_idx]
            
            if max_prob >= self.confidence_threshold:
                results.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'class': self.labels[max_idx],
                    'confidence': float(max_prob),
                    'all_probabilities': {self.labels[j]: float(probabilities[j]) for j in range(len(self.labels))}
                })
        
        return results
    
    def _onset_based_analysis(self, audio, sr):
        """온셋 기반 분석"""
        print("온셋 기반으로 분석 중...")
        
        # 온셋 감지
        onsets, rms = self.detect_onsets(audio, sr)
        
        results = []
        window_samples = int(self.window_size * sr)
        
        for onset_time in onsets:
            start_sample = int(onset_time * sr)
            end_sample = min(start_sample + window_samples, len(audio))
            
            if end_sample - start_sample < window_samples // 2:  # 너무 짧은 세그먼트는 스킵
                continue
            
            start_time = start_sample / sr
            end_time = end_sample / sr
            
            # 세그먼트 추출
            segment = audio[start_sample:end_sample]
            
            # 전처리
            processed = self.preprocess_segment(segment, sr)
            if processed is None:
                continue
            
            # 예측
            predictions = self.model.predict(processed, verbose=0)
            probabilities = predictions[0]
            
            # 최고 신뢰도 클래스 찾기
            max_idx = np.argmax(probabilities)
            max_prob = probabilities[max_idx]
            
            if max_prob >= self.confidence_threshold:
                results.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'class': self.labels[max_idx],
                    'confidence': float(max_prob),
                    'all_probabilities': {self.labels[j]: float(probabilities[j]) for j in range(len(self.labels))}
                })
        
        return results
    
    def visualize_results(self, results, duration, audio_path, save_path=None):
        """분석 결과 시각화"""
        if not results:
            print("시각화할 결과가 없습니다.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 타임라인 시각화
        ax1.set_title(f"드럼 분석 결과: {os.path.basename(audio_path)}", fontsize=14, fontweight='bold')
        
        # 클래스별 색상 매핑
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.labels)))
        class_colors = {label: colors[i] for i, label in enumerate(self.labels)}
        
        y_pos = 0
        for result in results:
            color = class_colors[result['class']]
            ax1.barh(y_pos, result['duration'], left=result['start_time'], 
                    height=0.8, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # 신뢰도 텍스트 추가
            mid_time = result['start_time'] + result['duration'] / 2
            ax1.text(mid_time, y_pos, f"{result['confidence']:.2f}", 
                    ha='center', va='center', fontsize=8, fontweight='bold')
            
            y_pos += 1
        
        ax1.set_xlim(0, duration)
        ax1.set_xlabel('시간 (초)')
        ax1.set_ylabel('세그먼트')
        ax1.set_title('드럼 사운드 타임라인')
        
        # 범례 추가
        legend_elements = [plt.Rectangle((0,0),1,1, color=class_colors[label], label=label) 
                          for label in set([r['class'] for r in results])]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # 2. 클래스별 분포
        class_counts = {}
        for result in results:
            class_name = result['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors_bar = [class_colors[cls] for cls in classes]
        
        ax2.bar(classes, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax2.set_title('클래스별 드럼 사운드 개수')
        ax2.set_xlabel('드럼 클래스')
        ax2.set_ylabel('개수')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"시각화 결과 저장: {save_path}")
        
        plt.show()
    
    def export_results(self, results, output_path):
        """분석 결과를 텍스트 파일로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("드럼 사운드 분석 결과\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"[{i}] {result['class']}\n")
                f.write(f"    시간: {result['start_time']:.2f}초 - {result['end_time']:.2f}초\n")
                f.write(f"    지속시간: {result['duration']:.2f}초\n")
                f.write(f"    신뢰도: {result['confidence']:.3f}\n")
                f.write(f"    전체 확률:\n")
                for class_name, prob in result['all_probabilities'].items():
                    f.write(f"      {class_name}: {prob:.3f}\n")
                f.write("\n")
        
        print(f"결과 파일 저장: {output_path}")

# =====================================================================
# ▼▼▼ 사용 예시 함수 ▼▼▼
# =====================================================================

def demo_continuous_analysis():
    """연속 음원 분석 데모"""
    MODEL_PATH = r"C:\GitHub\Music_Helper_Drum\models\final_crnn_drum_model.h5"
    TEST_SOUND_DIR = r"C:\GitHub\Music_Helper_Drum\TestSound"
    OUTPUT_DIR = r"C:\GitHub\Music_Helper_Drum\output"
    
    try:
        analyzer = ContinuousDrumAnalyzer(MODEL_PATH)
    except:
        print("모델 파일이 없습니다. 먼저 crnn_drum_classifier.py를 실행해서 모델을 훈련하세요.")
        return
    
    # TestSound 폴더의 모든 오디오 파일 찾기
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.aif', '.aiff']
    test_files = []
    
    if os.path.exists(TEST_SOUND_DIR):
        for file in os.listdir(TEST_SOUND_DIR):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                test_files.append(os.path.join(TEST_SOUND_DIR, file))
    
    if not test_files:
        print(f"TestSound 폴더에서 오디오 파일을 찾을 수 없습니다: {TEST_SOUND_DIR}")
        return
    
    print("="*60)
    print("연속 음원 드럼 분석 데모")
    print("="*60)
    print(f"\n총 {len(test_files)}개의 오디오 파일 발견")
    print("-" * 60)
    
    # 각 파일 분석
    for idx, test_audio in enumerate(test_files, 1):
        filename = os.path.basename(test_audio)
        print(f"\n[{idx}/{len(test_files)}] 분석 중: {filename}")
        print("=" * 60)
        
        try:
            # 슬라이딩 윈도우 방식으로 분석
            print("\n1. 슬라이딩 윈도우 방식 분석")
            results_sw, duration = analyzer.analyze_continuous_audio(test_audio, method='sliding_window')
            
            print(f"\n총 {len(results_sw)}개의 드럼 사운드 감지")
            for i, result in enumerate(results_sw[:10], 1):
                print(f"{i:2d}. {result['start_time']:6.2f}초 - {result['end_time']:6.2f}초: {result['class']} (신뢰도: {result['confidence']:.3f})")
            
            if len(results_sw) > 10:
                print(f"... 및 {len(results_sw) - 10}개 더")
            
            # 온셋 기반 분석
            print("\n2. 온셋 기반 분석")
            results_onset, _ = analyzer.analyze_continuous_audio(test_audio, method='onset_based')
            
            print(f"\n총 {len(results_onset)}개의 드럼 사운드 감지")
            for i, result in enumerate(results_onset[:10], 1):
                print(f"{i:2d}. {result['start_time']:6.2f}초 - {result['end_time']:6.2f}초: {result['class']} (신뢰도: {result['confidence']:.3f})")
            
            if len(results_onset) > 10:
                print(f"... 및 {len(results_onset) - 10}개 더")
            
            # 파일별 출력 경로 설정
            base_name = os.path.splitext(filename)[0]
            output_png = os.path.join(OUTPUT_DIR, f"{base_name}_continuous_sw.png")
            output_txt = os.path.join(OUTPUT_DIR, f"{base_name}_continuous_sw.txt")
            
            # 시각화
            print("\n3. 결과 시각화")
            analyzer.visualize_results(results_sw, duration, test_audio, save_path=output_png)
            
            # 결과 내보내기
            print("\n4. 결과 내보내기")
            analyzer.export_results(results_sw, output_txt)
            
            print(f"\n{filename} 분석 완료")
            print("=" * 60)
            
        except Exception as e:
            print(f"오류 발생: {e}")
            continue
    
    print("\n" + "="*60)
    print(f"전체 분석 완료! 총 {len(test_files)}개 파일 처리됨")
    print(f"결과 파일 위치: {OUTPUT_DIR}")
    print("="*60)

def interactive_continuous_analysis():
    """대화형 연속 음원 분석"""
    MODEL_PATH = r"C:\GitHub\Music_Helper_Drum\models\final_crnn_drum_model.h5"
    
    try:
        analyzer = ContinuousDrumAnalyzer(MODEL_PATH)
    except:
        print("모델 파일이 없습니다. 먼저 crnn_drum_classifier.py를 실행해서 모델을 훈련하세요.")
        return
    
    print("="*60)
    print("대화형 연속 음원 드럼 분석")
    print("종료하려면 'quit' 입력")
    print("="*60)
    
    while True:
        audio_path = input("\n분석할 오디오 파일 경로를 입력하세요: ").strip()
        
        if audio_path.lower() == 'quit':
            print("분석을 종료합니다.")
            break
        
        if not os.path.exists(audio_path):
            print("파일을 찾을 수 없습니다.")
            continue
        
        method = input("분석 방법을 선택하세요 (sliding_window/onset_based): ").strip().lower()
        if method not in ['sliding_window', 'onset_based']:
            method = 'sliding_window'
        
        print(f"\n{method} 방식으로 분석 중...")
        results, duration = analyzer.analyze_continuous_audio(audio_path, method=method)
        
        print(f"\n총 {len(results)}개의 드럼 사운드 감지")
        for i, result in enumerate(results, 1):
            print(f"{i:2d}. {result['start_time']:6.2f}초 - {result['end_time']:6.2f}초: {result['class']} (신뢰도: {result['confidence']:.3f})")
        
        # 시각화 여부 확인
        visualize = input("\n결과를 시각화하시겠습니까? (y/n): ").strip().lower()
        if visualize == 'y':
            analyzer.visualize_results(results, duration, audio_path)

if __name__ == "__main__":
    # 데모 실행
    demo_continuous_analysis()
    
    # 대화형 분석 (선택사항)
    # interactive_continuous_analysis()
