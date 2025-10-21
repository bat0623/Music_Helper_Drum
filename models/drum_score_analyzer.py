import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 드럼 악보 분석기 클래스
# =====================================================================

class DrumScoreAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.labels = ['bass_drum', 'crash_cymbal', 'high_tom', 'hihat_closed', 
                      'hihat_open', 'low_tom', 'mid_tom', 'ride_cymbal', 'snare_drum']
        
        # 한국어 라벨 매핑
        self.korean_labels = {
            'bass_drum': '베이스드럼',
            'crash_cymbal': '크래시심벌',
            'high_tom': '하이톰',
            'hihat_closed': '클로즈드하이햇',
            'hihat_open': '오픈하이햇',
            'low_tom': '로우톰',
            'mid_tom': '미드톰',
            'ride_cymbal': '라이드심벌',
            'snare_drum': '스네어드럼'
        }
        
        # 약어 매핑 (악보 표기용)
        self.abbreviations = {
            'bass_drum': 'BD',
            'crash_cymbal': 'CC',
            'high_tom': 'HT',
            'hihat_closed': 'HH',
            'hihat_open': 'OH',
            'low_tom': 'LT',
            'mid_tom': 'MT',
            'ride_cymbal': 'RC',
            'snare_drum': 'SD'
        }
        
        # 전처리 설정
        self.config = {
            "sr": 22050,
            "n_mels": 128,
            "n_fft": 2048,
            "hop_length": 512,
            "max_len": 200
        }
        
        # 분석 설정
        self.window_size = 0.5      # 윈도우 크기 (초)
        self.hop_size = 0.05        # 윈도우 이동 간격 (초)
        self.confidence_threshold = 0.5  # 신뢰도 임계값
        
        self.load_model()
    
    def load_model(self):
        """훈련된 모델 로딩"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"모델 로딩 성공: {os.path.basename(self.model_path)}")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise
    
    def preprocess_segment(self, audio_segment, sr):
        """오디오 세그먼트 전처리"""
        try:
            # 너무 짧은 세그먼트는 패딩
            min_length = int(0.1 * sr)
            if len(audio_segment) < min_length:
                audio_segment = np.pad(audio_segment, (0, min_length - len(audio_segment)), mode='constant')
            
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
            return None
    
    def analyze_continuous_audio(self, audio_path):
        """연속 음원 분석"""
        print(f"\n분석 시작: {os.path.basename(audio_path)}")
        
        # 오디오 로딩
        audio, sr = librosa.load(audio_path, sr=self.config["sr"])
        duration = len(audio) / sr
        
        print(f"오디오 길이: {duration:.2f}초")
        print(f"샘플레이트: {sr}Hz")
        print(f"분석 중...")
        
        # 슬라이딩 윈도우로 전체 분석
        results = []
        window_samples = int(self.window_size * sr)
        hop_samples = int(self.hop_size * sr)
        
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
            
            # 상위 3개 클래스 저장
            top3_indices = np.argsort(probabilities)[::-1][:3]
            
            for rank, idx in enumerate(top3_indices):
                prob = probabilities[idx]
                if prob >= self.confidence_threshold:
                    results.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'center_time': (start_time + end_time) / 2,
                        'class': self.labels[idx],
                        'confidence': float(prob),
                        'rank': rank + 1
                    })
        
        print(f"분석 완료: 총 {len(results)}개의 드럼 사운드 후보 감지")
        
        # 후처리: 중복 제거 및 정리
        results = self._post_process_results(results)
        
        print(f"후처리 완료: {len(results)}개의 드럼 사운드 확정")
        
        return results, duration
    
    def _post_process_results(self, results):
        """결과 후처리 - 중복 제거 및 정리"""
        if not results:
            return results
        
        # 시간순 정렬
        results.sort(key=lambda x: x['center_time'])
        
        # 시간 기준으로 그룹화 (0.2초 이내의 같은 클래스는 하나로)
        filtered_results = []
        time_threshold = 0.2
        
        i = 0
        while i < len(results):
            current = results[i]
            
            # 같은 시간대의 같은 클래스 찾기
            group = [current]
            j = i + 1
            
            while j < len(results):
                if (results[j]['center_time'] - current['center_time'] < time_threshold and
                    results[j]['class'] == current['class']):
                    group.append(results[j])
                    j += 1
                else:
                    break
            
            # 그룹 중 가장 높은 신뢰도를 가진 것만 선택
            best = max(group, key=lambda x: x['confidence'])
            filtered_results.append(best)
            
            i = j if j > i + 1 else i + 1
        
        # 다시 시간순 정렬
        filtered_results.sort(key=lambda x: x['start_time'])
        
        return filtered_results
    
    def print_score_timeline(self, results, duration):
        """악보 형식의 타임라인 출력"""
        if not results:
            print("\n감지된 드럼 사운드가 없습니다.")
            return
        
        print("\n" + "="*80)
        print("드럼 악보 분석 결과")
        print("="*80)
        
        # 통계
        print(f"\n[분석 통계]")
        print(f"총 음원 길이: {duration:.2f}초")
        print(f"감지된 드럼 사운드: {len(results)}개")
        
        # 클래스별 통계
        class_counts = {}
        for result in results:
            class_name = result['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n[클래스별 분포]")
        for class_name in sorted(class_counts.keys()):
            korean_name = self.korean_labels[class_name]
            abbr = self.abbreviations[class_name]
            count = class_counts[class_name]
            print(f"  {abbr} ({korean_name}): {count}개")
        
        # 악보 형식 타임라인
        print(f"\n[타임라인 - 악보 형식]")
        print("-" * 80)
        print(f"{'시간(초)':<10} | {'드럼 종류':<20} | {'신뢰도':<10}")
        print("-" * 80)
        
        for result in results:
            time_str = f"{result['start_time']:6.2f}"
            abbr = self.abbreviations[result['class']]
            korean_name = self.korean_labels[result['class']]
            drum_str = f"{abbr} ({korean_name})"
            confidence_str = f"{result['confidence']*100:5.1f}%"
            
            print(f"{time_str:<10} | {drum_str:<20} | {confidence_str:<10}")
        
        # 0.1초 단위 타임라인 (간격별 표시)
        print(f"\n[상세 타임라인 - 0.1초 단위]")
        print("-" * 80)
        
        time_step = 0.1
        current_time = 0.0
        
        while current_time <= duration:
            time_str = f"{current_time:5.2f}s"
            
            # 현재 시간에서 +/- 0.05초 이내의 드럼 찾기
            active_drums = []
            for result in results:
                if abs(result['center_time'] - current_time) <= 0.05:
                    abbr = self.abbreviations[result['class']]
                    active_drums.append(abbr)
            
            if active_drums:
                drums_str = " ".join(active_drums)
                print(f"{time_str} | {drums_str}")
            else:
                print(f"{time_str} |")
            
            current_time += time_step
        
        print("-" * 80)
    
    def print_detailed_results(self, results):
        """상세 분석 결과 출력"""
        if not results:
            return
        
        print(f"\n[상세 분석 결과]")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            abbr = self.abbreviations[result['class']]
            korean_name = self.korean_labels[result['class']]
            
            print(f"\n{i}. {abbr} - {korean_name}")
            print(f"   시간: {result['start_time']:.3f}초 ~ {result['end_time']:.3f}초")
            print(f"   중심: {result['center_time']:.3f}초")
            print(f"   신뢰도: {result['confidence']*100:.1f}%")
    
    def create_score_visualization(self, results, duration, audio_path, save_path=None):
        """악보 스타일 시각화"""
        if not results:
            print("시각화할 결과가 없습니다.")
            return
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 타임라인만 크게 표시
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # 타이틀
        ax.set_title(f"드럼 악보 타임라인: {os.path.basename(audio_path)}", 
                     fontsize=18, fontweight='bold', pad=20)
        
        # 클래스별 y축 위치 고정
        class_y_positions = {label: i for i, label in enumerate(self.labels)}
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.labels)))
        class_colors = {label: colors[i] for i, label in enumerate(self.labels)}
        
        # 각 드럼 사운드를 점으로 표시
        for result in results:
            class_name = result['class']
            y_pos = class_y_positions[class_name]
            color = class_colors[class_name]
            
            # 마커로 표시 (크기는 신뢰도에 비례)
            size = result['confidence'] * 500
            ax.scatter(result['center_time'], y_pos, s=size, c=[color], 
                       alpha=0.7, edgecolors='black', linewidth=2)
            
            # 시간 표시
            ax.text(result['center_time'], y_pos + 0.35, 
                    f"{result['center_time']:.2f}s", 
                    ha='center', fontsize=10, alpha=0.8, fontweight='bold')
            
            # 신뢰도 표시
            ax.text(result['center_time'], y_pos - 0.35, 
                    f"{result['confidence']*100:.0f}%", 
                    ha='center', fontsize=9, alpha=0.6)
        
        # 축 설정
        ax.set_xlim(-0.1, duration + 0.1)
        ax.set_ylim(-0.5, len(self.labels) - 0.5)
        ax.set_xlabel('시간 (초)', fontsize=14, fontweight='bold')
        ax.set_ylabel('드럼 종류', fontsize=14, fontweight='bold')
        
        # y축 레이블
        ax.set_yticks(range(len(self.labels)))
        ax.set_yticklabels([f"{self.abbreviations[label]} ({self.korean_labels[label]})" 
                             for label in self.labels], fontsize=12)
        
        # 그리드 설정
        ax.grid(True, alpha=0.3, axis='x', linewidth=0.8)
        ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)
        
        # 상하 경계선
        ax.axhline(y=-0.5, color='black', linewidth=2)
        ax.axhline(y=len(self.labels)-0.5, color='black', linewidth=2)
        
        # x축 눈금 간격 설정 (0.1초 단위)
        x_major_ticks = np.arange(0, duration + 0.1, 0.5)
        x_minor_ticks = np.arange(0, duration + 0.1, 0.1)
        ax.set_xticks(x_major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.tick_params(axis='x', labelsize=11)
        
        # 범례 추가
        legend_elements = []
        for label in self.labels:
            if any(r['class'] == label for r in results):
                legend_elements.append(
                    plt.scatter([], [], s=200, c=[class_colors[label]], 
                              alpha=0.7, edgecolors='black', linewidth=2,
                              label=f"{self.abbreviations[label]} ({self.korean_labels[label]})")
                )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1.12, 1), fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"시각화 결과 저장: {save_path}")
        
        plt.show()
    
    def export_results(self, results, duration, output_path):
        """분석 결과를 텍스트 파일로 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("드럼 악보 분석 결과\n")
            f.write("=" * 60 + "\n\n")
            
            # 통계
            f.write("[분석 통계]\n")
            f.write(f"총 음원 길이: {duration:.2f}초\n")
            f.write(f"감지된 드럼 사운드: {len(results)}개\n\n")
            
            # 클래스별 통계
            class_counts = {}
            for result in results:
                class_name = result['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            f.write("[클래스별 분포]\n")
            for class_name in sorted(class_counts.keys()):
                korean_name = self.korean_labels[class_name]
                abbr = self.abbreviations[class_name]
                count = class_counts[class_name]
                f.write(f"  {abbr} ({korean_name}): {count}개\n")
            
            # 타임라인
            f.write(f"\n[타임라인]\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'시간(초)':<10} | {'드럼 종류':<20} | {'신뢰도':<10}\n")
            f.write("-" * 60 + "\n")
            
            for result in results:
                time_str = f"{result['start_time']:6.2f}"
                abbr = self.abbreviations[result['class']]
                korean_name = self.korean_labels[result['class']]
                drum_str = f"{abbr} ({korean_name})"
                confidence_str = f"{result['confidence']*100:5.1f}%"
                
                f.write(f"{time_str:<10} | {drum_str:<20} | {confidence_str:<10}\n")
            
            # 상세 결과
            f.write(f"\n[상세 분석 결과]\n")
            f.write("=" * 60 + "\n")
            
            for i, result in enumerate(results, 1):
                abbr = self.abbreviations[result['class']]
                korean_name = self.korean_labels[result['class']]
                
                f.write(f"\n{i}. {abbr} - {korean_name}\n")
                f.write(f"   시간: {result['start_time']:.3f}초 ~ {result['end_time']:.3f}초\n")
                f.write(f"   중심: {result['center_time']:.3f}초\n")
                f.write(f"   신뢰도: {result['confidence']*100:.1f}%\n")
        
        print(f"결과 파일 저장: {output_path}")

# =====================================================================
# 사용 예시 함수
# =====================================================================

def demo_score_analysis():
    """드럼 악보 분석 데모"""
    print("="*60)
    print("드럼 악보 분석 시스템 데모")
    print("="*60)
    
    MODEL_PATH = r"C:\GitHub\Music_Helper_Drum\models\final_crnn_drum_model.h5"
    TEST_SOUND_DIR = r"C:\GitHub\Music_Helper_Drum\TestSound"
    OUTPUT_DIR = r"C:\GitHub\Music_Helper_Drum\output"
    
    try:
        analyzer = DrumScoreAnalyzer(MODEL_PATH)
    except Exception as e:
        print(f"분석기 초기화 실패: {e}")
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
    
    print(f"\n총 {len(test_files)}개의 오디오 파일 발견")
    print("-" * 60)
    
    # 각 파일 분석
    for idx, test_audio in enumerate(test_files, 1):
        filename = os.path.basename(test_audio)
        print(f"\n[{idx}/{len(test_files)}] 분석 중: {filename}")
        print("=" * 60)
        
        try:
            # 분석 실행
            results, duration = analyzer.analyze_continuous_audio(test_audio)
            
            # 결과 출력
            analyzer.print_score_timeline(results, duration)
            analyzer.print_detailed_results(results)
            
            # 파일별 출력 경로 설정
            base_name = os.path.splitext(filename)[0]
            output_png = os.path.join(OUTPUT_DIR, f"{base_name}_analysis.png")
            output_txt = os.path.join(OUTPUT_DIR, f"{base_name}_analysis.txt")
            
            # 시각화
            print(f"\n시각화 생성 중...")
            analyzer.create_score_visualization(results, duration, test_audio, 
                                               save_path=output_png)
            
            # 결과 내보내기
            print(f"결과 내보내기...")
            analyzer.export_results(results, duration, output_txt)
            
            print(f"\n{filename} 분석 완료")
            print("=" * 60)
            
        except Exception as e:
            print(f"오류 발생: {e}")
            continue
    
    print("\n" + "="*60)
    print(f"전체 분석 완료! 총 {len(test_files)}개 파일 처리됨")
    print(f"결과 파일 위치: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    demo_score_analysis()
