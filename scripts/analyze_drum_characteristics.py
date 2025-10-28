import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =====================================================================
# 드럼 샘플 특징 분석 스크립트
# =====================================================================

# 프로젝트 루트 경로 자동 감지
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 원본 데이터 경로
DATA_PATH = os.path.join(PROJECT_ROOT, "drum_samples")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "debug_img", "drum_analysis")

# 결과 저장 폴더 생성
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 오디오 설정
SR = 22050

# 드럼 클래스
labels = ['bass_drum', 'crash_cymbal', 'high_tom', 'hihat_closed', 
          'hihat_open', 'low_tom', 'mid_tom', 'ride_cymbal', 'snare_drum']

print("="*70)
print("드럼 샘플 특징 분석 시작")
print("="*70)

# 각 클래스별 분석 결과 저장
analysis_results = {}

for label in labels:
    label_path = os.path.join(DATA_PATH, label)
    wav_files = [f for f in os.listdir(label_path) if f.endswith('.wav')]
    
    print(f"\n{'='*70}")
    print(f"[분석] {label} 분석 중... ({len(wav_files)}개 파일)")
    print('='*70)
    
    # 분석할 특징들
    durations = []
    energy_durations = []  # 실제 신호가 있는 구간
    rms_energies = []
    peak_amplitudes = []
    spectral_centroids = []
    zero_crossing_rates = []
    onset_strengths = []
    
    # 샘플 3개만 시각화 (대표 샘플)
    sample_files = wav_files[:3]
    
    for wav_file in wav_files:
        try:
            file_path = os.path.join(label_path, wav_file)
            y, sr = librosa.load(file_path, sr=SR)
            
            # 1. 전체 길이
            duration = librosa.get_duration(y=y, sr=sr)
            durations.append(duration)
            
            # 2. 에너지 기반 실제 신호 길이 (40dB threshold)
            y_trimmed, index = librosa.effects.trim(y, top_db=40)
            energy_duration = librosa.get_duration(y=y_trimmed, sr=sr)
            energy_durations.append(energy_duration)
            
            # 3. RMS 에너지
            rms = librosa.feature.rms(y=y)[0]
            rms_energies.append(np.mean(rms))
            
            # 4. Peak Amplitude
            peak_amp = np.max(np.abs(y))
            peak_amplitudes.append(peak_amp)
            
            # 5. Spectral Centroid (주파수 중심)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroids.append(np.mean(spectral_centroid))
            
            # 6. Zero Crossing Rate (음색 특성)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zero_crossing_rates.append(np.mean(zcr))
            
            # 7. Onset Strength (타격 강도)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_strengths.append(np.max(onset_env))
            
        except Exception as e:
            print(f"  [WARNING] {wav_file} 분석 실패: {e}")
    
    # 통계 계산
    result = {
        'count': len(durations),
        'duration': {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations)
        },
        'energy_duration': {
            'mean': np.mean(energy_durations),
            'std': np.std(energy_durations),
            'min': np.min(energy_durations),
            'max': np.max(energy_durations),
            'percentile_95': np.percentile(energy_durations, 95)
        },
        'rms_energy': {
            'mean': np.mean(rms_energies),
            'std': np.std(rms_energies)
        },
        'peak_amplitude': {
            'mean': np.mean(peak_amplitudes),
            'std': np.std(peak_amplitudes)
        },
        'spectral_centroid': {
            'mean': np.mean(spectral_centroids),
            'std': np.std(spectral_centroids)
        },
        'zero_crossing_rate': {
            'mean': np.mean(zero_crossing_rates),
            'std': np.std(zero_crossing_rates)
        },
        'onset_strength': {
            'mean': np.mean(onset_strengths),
            'std': np.std(onset_strengths)
        }
    }
    
    analysis_results[label] = result
    
    # 결과 출력
    print(f"\n[결과] {label} 분석 결과:")
    print(f"  파일 수: {result['count']}개")
    print(f"\n  [길이 정보]")
    print(f"    전체 길이: {result['duration']['mean']:.3f}초 (±{result['duration']['std']:.3f})")
    print(f"    범위: {result['duration']['min']:.3f}초 ~ {result['duration']['max']:.3f}초")
    print(f"    실제 신호 길이: {result['energy_duration']['mean']:.3f}초 (±{result['energy_duration']['std']:.3f})")
    print(f"    실제 신호 95%: {result['energy_duration']['percentile_95']:.3f}초")
    print(f"\n  [음향 특성]")
    print(f"    RMS 에너지: {result['rms_energy']['mean']:.4f}")
    print(f"    Peak 진폭: {result['peak_amplitude']['mean']:.4f}")
    print(f"    주파수 중심: {result['spectral_centroid']['mean']:.0f} Hz")
    print(f"    Zero Crossing Rate: {result['zero_crossing_rate']['mean']:.4f}")
    print(f"    타격 강도: {result['onset_strength']['mean']:.4f}")
    
    # 대표 샘플 3개 시각화
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'{label} - 대표 샘플 분석', fontsize=16, fontweight='bold')
    
    for idx, wav_file in enumerate(sample_files):
        try:
            file_path = os.path.join(label_path, wav_file)
            y, sr = librosa.load(file_path, sr=SR)
            
            # 원본 파형
            ax1 = axes[idx, 0]
            times = np.arange(len(y)) / sr
            ax1.plot(times, y, linewidth=0.5)
            ax1.set_title(f'샘플 {idx+1}: 파형')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            
            # Trimmed 파형 (에너지 기반)
            ax2 = axes[idx, 1]
            y_trimmed, _ = librosa.effects.trim(y, top_db=40)
            times_trimmed = np.arange(len(y_trimmed)) / sr
            ax2.plot(times_trimmed, y_trimmed, linewidth=0.5, color='green')
            ax2.set_title(f'Trimmed (40dB): {len(y_trimmed)/sr:.3f}s')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            
            # 멜 스펙트로그램
            ax3 = axes[idx, 2]
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            melspec_db = librosa.power_to_db(melspec, ref=np.max)
            img = ax3.imshow(melspec_db, aspect='auto', origin='lower', cmap='viridis')
            ax3.set_title(f'Mel Spectrogram')
            ax3.set_xlabel('Time Frames')
            ax3.set_ylabel('Mel Bins')
            plt.colorbar(img, ax=ax3, format='%+2.0f dB')
            
        except Exception as e:
            print(f"  [WARNING] {wav_file} 시각화 실패: {e}")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_PATH, f'{label}_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [저장] 시각화 저장: {save_path}")

# =====================================================================
# 종합 비교 분석
# =====================================================================

print(f"\n{'='*70}")
print("[비교] 클래스별 종합 비교")
print('='*70)

# 비교 테이블 출력
print(f"\n{'클래스':<15} {'실제신호(초)':<12} {'95%길이(초)':<12} {'주파수중심(Hz)':<15} {'권장길이(초)'}")
print('-'*70)

recommendations = {}
for label in labels:
    result = analysis_results[label]
    energy_mean = result['energy_duration']['mean']
    energy_95 = result['energy_duration']['percentile_95']
    spectral = result['spectral_centroid']['mean']
    
    # 권장 길이 계산 (95% + 여유)
    recommended = min(energy_95 * 1.2, 2.0)  # 최대 2초
    recommendations[label] = recommended
    
    print(f"{label:<15} {energy_mean:<12.3f} {energy_95:<12.3f} {spectral:<15.0f} {recommended:.3f}")

# 시각화: 클래스별 비교
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 실제 신호 길이 비교
ax1 = axes[0, 0]
energy_means = [analysis_results[l]['energy_duration']['mean'] for l in labels]
energy_stds = [analysis_results[l]['energy_duration']['std'] for l in labels]
x_pos = np.arange(len(labels))
ax1.bar(x_pos, energy_means, yerr=energy_stds, capsize=5, alpha=0.7, color='steelblue')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.set_ylabel('Duration (seconds)')
ax1.set_title('실제 신호 길이 비교 (40dB Trim)')
ax1.grid(True, alpha=0.3, axis='y')

# 2. 주파수 중심 비교
ax2 = axes[0, 1]
spectral_means = [analysis_results[l]['spectral_centroid']['mean'] for l in labels]
colors = ['red' if s < 2000 else 'orange' if s < 5000 else 'green' for s in spectral_means]
ax2.bar(x_pos, spectral_means, color=colors, alpha=0.7)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=45, ha='right')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('주파수 중심 비교 (저음/중음/고음)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=2000, color='red', linestyle='--', alpha=0.5, label='Low')
ax2.axhline(y=5000, color='orange', linestyle='--', alpha=0.5, label='Mid')
ax2.legend()

# 3. Zero Crossing Rate 비교
ax3 = axes[1, 0]
zcr_means = [analysis_results[l]['zero_crossing_rate']['mean'] for l in labels]
ax3.bar(x_pos, zcr_means, alpha=0.7, color='purple')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, rotation=45, ha='right')
ax3.set_ylabel('Zero Crossing Rate')
ax3.set_title('Zero Crossing Rate 비교 (음색 특성)')
ax3.grid(True, alpha=0.3, axis='y')

# 4. 권장 최대 길이
ax4 = axes[1, 1]
recommended_values = [recommendations[l] for l in labels]
ax4.bar(x_pos, recommended_values, alpha=0.7, color='green')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(labels, rotation=45, ha='right')
ax4.set_ylabel('Duration (seconds)')
ax4.set_title('권장 최대 길이 (95% + 20% 여유)')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
comparison_path = os.path.join(OUTPUT_PATH, 'class_comparison.png')
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[저장] 종합 비교 저장: {comparison_path}")

# =====================================================================
# 권장 파라미터 출력
# =====================================================================

print(f"\n{'='*70}")
print("[권장] 증강 코드에 적용할 권장 파라미터")
print('='*70)

print("\n# 클래스별 최대 길이 설정 (초)")
print("CLASS_MAX_DURATION = {")
for label in labels:
    print(f"    '{label}': {recommendations[label]:.3f},")
print("}")

print("\n# 클래스별 Trim Threshold (dB)")
print("CLASS_TRIM_THRESHOLD = {")
for label in labels:
    result = analysis_results[label]
    # 에너지가 높은 클래스는 더 엄격한 threshold
    if result['rms_energy']['mean'] > 0.1:
        threshold = 35
    elif result['rms_energy']['mean'] > 0.05:
        threshold = 40
    else:
        threshold = 45
    print(f"    '{label}': {threshold},  # RMS: {result['rms_energy']['mean']:.4f}")
print("}")

print("\n" + "="*70)
print("[완료] 분석 완료!")
print(f"[경로] 결과 저장 위치: {OUTPUT_PATH}")
print("="*70)

