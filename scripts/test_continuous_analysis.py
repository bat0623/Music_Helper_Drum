#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연속 음원 드럼 분석 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from continuous_drum_analyzer import ContinuousDrumAnalyzer
import matplotlib.pyplot as plt

def test_with_sample_audio():
    """샘플 오디오로 테스트"""
    print("="*60)
    print("연속 음원 드럼 분석 테스트")
    print("="*60)
    
    # 모델 경로
    MODEL_PATH = r"C:\GitHub\Music_Helper_Drum\models\final_crnn_drum_model.h5"
    
    # 분석기 초기화
    try:
        analyzer = ContinuousDrumAnalyzer(MODEL_PATH)
        print("✅ 분석기 초기화 성공")
    except Exception as e:
        print(f"❌ 분석기 초기화 실패: {e}")
        return
    
    # TestSound 폴더의 모든 오디오 파일 찾기
    TEST_SOUND_DIR = r"C:\GitHub\Music_Helper_Drum\TestSound"
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
    
    for audio_path in test_files:
        if not os.path.exists(audio_path):
            print(f"⚠️ 파일을 찾을 수 없습니다: {audio_path}")
            continue
        
        print(f"\n🎵 분석 중: {os.path.basename(audio_path)}")
        print("-" * 50)
        
        # 1. 슬라이딩 윈도우 방식 분석
        print("1️⃣ 슬라이딩 윈도우 방식 분석")
        try:
            results_sw, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
            print(f"   📊 총 {len(results_sw)}개의 드럼 사운드 감지")
            
            # 상위 5개 결과 출력
            for i, result in enumerate(results_sw[:5], 1):
                print(f"   {i}. {result['start_time']:6.2f}초 - {result['end_time']:6.2f}초: {result['class']} (신뢰도: {result['confidence']:.3f})")
            
            if len(results_sw) > 5:
                print(f"   ... 및 {len(results_sw) - 5}개 더")
                
        except Exception as e:
            print(f"   ❌ 슬라이딩 윈도우 분석 실패: {e}")
            results_sw = []
            duration = 0
        
        # 2. 온셋 기반 분석
        print("\n2️⃣ 온셋 기반 분석")
        try:
            results_onset, _ = analyzer.analyze_continuous_audio(audio_path, method='onset_based')
            print(f"   📊 총 {len(results_onset)}개의 드럼 사운드 감지")
            
            # 상위 5개 결과 출력
            for i, result in enumerate(results_onset[:5], 1):
                print(f"   {i}. {result['start_time']:6.2f}초 - {result['end_time']:6.2f}초: {result['class']} (신뢰도: {result['confidence']:.3f})")
            
            if len(results_onset) > 5:
                print(f"   ... 및 {len(results_onset) - 5}개 더")
                
        except Exception as e:
            print(f"   ❌ 온셋 기반 분석 실패: {e}")
            results_onset = []
        
        # 3. 결과 비교
        print(f"\n📈 분석 방법 비교:")
        print(f"   슬라이딩 윈도우: {len(results_sw)}개 감지")
        print(f"   온셋 기반: {len(results_onset)}개 감지")
        
        # 4. 시각화 (결과가 있는 경우에만)
        if results_sw or results_onset:
            print(f"\n3️⃣ 결과 시각화")
            try:
                # 슬라이딩 윈도우 결과로 시각화
                if results_sw:
                    analyzer.visualize_results(results_sw, duration, audio_path, 
                                            save_path=r"C:\GitHub\Music_Helper_Drum\output\continuous_analysis_sw.png")
                    print("   ✅ 슬라이딩 윈도우 결과 시각화 완료")
                
                # 온셋 기반 결과로 시각화
                if results_onset:
                    analyzer.visualize_results(results_onset, duration, audio_path, 
                                            save_path=r"C:\GitHub\Music_Helper_Drum\output\continuous_analysis_onset.png")
                    print("   ✅ 온셋 기반 결과 시각화 완료")
                    
            except Exception as e:
                print(f"   ❌ 시각화 실패: {e}")
        
        # 5. 결과 내보내기
        print(f"\n4️⃣ 결과 내보내기")
        try:
            if results_sw:
                analyzer.export_results(results_sw, r"C:\GitHub\Music_Helper_Drum\output\continuous_analysis_sw_results.txt")
                print("   ✅ 슬라이딩 윈도우 결과 파일 저장 완료")
            
            if results_onset:
                analyzer.export_results(results_onset, r"C:\GitHub\Music_Helper_Drum\output\continuous_analysis_onset_results.txt")
                print("   ✅ 온셋 기반 결과 파일 저장 완료")
                
        except Exception as e:
            print(f"   ❌ 결과 내보내기 실패: {e}")

def test_parameter_adjustment():
    """파라미터 조정 테스트"""
    print("\n" + "="*60)
    print("파라미터 조정 테스트")
    print("="*60)
    
    MODEL_PATH = r"C:\GitHub\Music_Helper_Drum\models\final_crnn_drum_model.h5"
    audio_path = r"C:\GitHub\Music_Helper_Drum\models\Untitled.wav"
    
    if not os.path.exists(audio_path):
        print(f"⚠️ 테스트 파일을 찾을 수 없습니다: {audio_path}")
        return
    
    try:
        analyzer = ContinuousDrumAnalyzer(MODEL_PATH)
        
        # 다른 파라미터로 테스트
        print("🔧 파라미터 조정 테스트")
        
        # 1. 더 작은 윈도우 크기
        print("\n1️⃣ 작은 윈도우 크기 (1초)")
        analyzer.window_size = 1.0
        analyzer.hop_size = 0.25
        results_small, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
        print(f"   📊 {len(results_small)}개 감지")
        
        # 2. 더 큰 윈도우 크기
        print("\n2️⃣ 큰 윈도우 크기 (3초)")
        analyzer.window_size = 3.0
        analyzer.hop_size = 1.0
        results_large, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
        print(f"   📊 {len(results_large)}개 감지")
        
        # 3. 낮은 신뢰도 임계값
        print("\n3️⃣ 낮은 신뢰도 임계값 (0.5)")
        analyzer.confidence_threshold = 0.5
        analyzer.window_size = 2.0
        analyzer.hop_size = 0.5
        results_low_conf, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
        print(f"   📊 {len(results_low_conf)}개 감지")
        
        # 4. 높은 신뢰도 임계값
        print("\n4️⃣ 높은 신뢰도 임계값 (0.9)")
        analyzer.confidence_threshold = 0.9
        results_high_conf, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
        print(f"   📊 {len(results_high_conf)}개 감지")
        
        print(f"\n📊 파라미터별 결과 비교:")
        print(f"   작은 윈도우 (1초): {len(results_small)}개")
        print(f"   큰 윈도우 (3초): {len(results_large)}개")
        print(f"   낮은 신뢰도 (0.5): {len(results_low_conf)}개")
        print(f"   높은 신뢰도 (0.9): {len(results_high_conf)}개")
        
    except Exception as e:
        print(f"❌ 파라미터 테스트 실패: {e}")

if __name__ == "__main__":
    # 기본 테스트
    test_with_sample_audio()
    
    # 파라미터 조정 테스트
    test_parameter_adjustment()
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("결과 파일들은 output 폴더에서 확인할 수 있습니다.")
    print("="*60)


