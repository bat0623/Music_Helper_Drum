#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연속 음원 드럼 분석 테스트 스크립트
"""

import sys
import os
import importlib.util
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# 숫자로 시작하는 모듈 import를 위한 특수 처리
spec = importlib.util.spec_from_file_location("continuous_analyzer", 
    os.path.join(os.path.dirname(__file__), '..', 'models', '3_continuous_analyzer.py'))
continuous_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuous_analyzer)
ContinuousDrumAnalyzer = continuous_analyzer.ContinuousDrumAnalyzer

import matplotlib.pyplot as plt

def test_with_sample_audio():
    """샘플 오디오로 테스트"""
    print("="*60)
    print("연속 음원 드럼 분석 테스트")
    print("="*60)
    
    # 프로젝트 루트 경로 자동 감지
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 모델 경로
    MODEL_PATH = os.path.join(project_root, "models", "final_crnn_drum_model.h5")
    
    # 분석기 초기화
    try:
        analyzer = ContinuousDrumAnalyzer(MODEL_PATH)
        print("[성공] 분석기 초기화 성공")
    except Exception as e:
        print(f"[실패] 분석기 초기화 실패: {e}")
        return
    
    # TestSound 폴더의 모든 오디오 파일 찾기
    TEST_SOUND_DIR = os.path.join(project_root, "TestSound")
    OUTPUT_DIR = os.path.join(project_root, "output")
    
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
            print(f"[경고] 파일을 찾을 수 없습니다: {audio_path}")
            continue
        
        print(f"\n[분석] {os.path.basename(audio_path)}")
        print("-" * 50)
        
        # 1. 슬라이딩 윈도우 방식 분석
        print("[1] 슬라이딩 윈도우 방식 분석")
        try:
            results_sw, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
            print(f"    총 {len(results_sw)}개의 드럼 사운드 감지")
            
            # 상위 5개 결과 출력
            for i, result in enumerate(results_sw[:5], 1):
                print(f"    {i}. {result['start_time']:6.2f}초 - {result['end_time']:6.2f}초: {result['class']} (신뢰도: {result['confidence']:.3f})")
            
            if len(results_sw) > 5:
                print(f"    ... 및 {len(results_sw) - 5}개 더")
                
        except Exception as e:
            print(f"    [실패] 슬라이딩 윈도우 분석 실패: {e}")
            results_sw = []
            duration = 0
        
        # 2. 온셋 기반 분석
        print("\n[2] 온셋 기반 분석")
        try:
            results_onset, _ = analyzer.analyze_continuous_audio(audio_path, method='onset_based')
            print(f"    총 {len(results_onset)}개의 드럼 사운드 감지")
            
            # 상위 5개 결과 출력
            for i, result in enumerate(results_onset[:5], 1):
                print(f"    {i}. {result['start_time']:6.2f}초 - {result['end_time']:6.2f}초: {result['class']} (신뢰도: {result['confidence']:.3f})")
            
            if len(results_onset) > 5:
                print(f"    ... 및 {len(results_onset) - 5}개 더")
                
        except Exception as e:
            print(f"    [실패] 온셋 기반 분석 실패: {e}")
            results_onset = []
        
        # 3. 결과 비교
        print(f"\n[비교] 분석 방법 비교:")
        print(f"    슬라이딩 윈도우: {len(results_sw)}개 감지")
        print(f"    온셋 기반: {len(results_onset)}개 감지")
        
        # 4. 시각화 (결과가 있는 경우에만)
        if results_sw or results_onset:
            print(f"\n[3] 결과 시각화")
            try:
                # 슬라이딩 윈도우 결과로 시각화
                if results_sw:
                    save_path = os.path.join(OUTPUT_DIR, "continuous_analysis_sw.png")
                    analyzer.visualize_results(results_sw, duration, audio_path, save_path=save_path)
                    print(f"    [성공] 슬라이딩 윈도우 결과 시각화 완료")
                
                # 온셋 기반 결과로 시각화
                if results_onset:
                    save_path = os.path.join(OUTPUT_DIR, "continuous_analysis_onset.png")
                    analyzer.visualize_results(results_onset, duration, audio_path, save_path=save_path)
                    print(f"    [성공] 온셋 기반 결과 시각화 완료")
                    
            except Exception as e:
                print(f"    [실패] 시각화 실패: {e}")
        
        # 5. 결과 내보내기
        print(f"\n[4] 결과 내보내기")
        try:
            if results_sw:
                save_path = os.path.join(OUTPUT_DIR, "continuous_analysis_sw_results.txt")
                analyzer.export_results(results_sw, save_path)
                print(f"    [성공] 슬라이딩 윈도우 결과 파일 저장 완료")
            
            if results_onset:
                save_path = os.path.join(OUTPUT_DIR, "continuous_analysis_onset_results.txt")
                analyzer.export_results(results_onset, save_path)
                print(f"    [성공] 온셋 기반 결과 파일 저장 완료")
                
        except Exception as e:
            print(f"    [실패] 결과 내보내기 실패: {e}")

def test_parameter_adjustment():
    """파라미터 조정 테스트"""
    print("\n" + "="*60)
    print("파라미터 조정 테스트")
    print("="*60)
    
    # 프로젝트 루트 경로 자동 감지
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    MODEL_PATH = os.path.join(project_root, "models", "final_crnn_drum_model.h5")
    TEST_SOUND_DIR = os.path.join(project_root, "TestSound")
    
    # 테스트 파일 찾기
    audio_path = None
    if os.path.exists(TEST_SOUND_DIR):
        for file in os.listdir(TEST_SOUND_DIR):
            if file.lower().endswith('.wav'):
                audio_path = os.path.join(TEST_SOUND_DIR, file)
                break
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"[경고] 테스트 파일을 찾을 수 없습니다: {TEST_SOUND_DIR}")
        return
    
    try:
        analyzer = ContinuousDrumAnalyzer(MODEL_PATH)
        
        # 다른 파라미터로 테스트
        print("[테스트] 파라미터 조정 테스트")
        
        # 1. 더 작은 윈도우 크기
        print("\n[1] 작은 윈도우 크기 (1초)")
        analyzer.window_size = 1.0
        analyzer.hop_size = 0.25
        results_small, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
        print(f"    {len(results_small)}개 감지")
        
        # 2. 더 큰 윈도우 크기
        print("\n[2] 큰 윈도우 크기 (3초)")
        analyzer.window_size = 3.0
        analyzer.hop_size = 1.0
        results_large, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
        print(f"    {len(results_large)}개 감지")
        
        # 3. 낮은 신뢰도 임계값
        print("\n[3] 낮은 신뢰도 임계값 (0.5)")
        analyzer.confidence_threshold = 0.5
        analyzer.window_size = 2.0
        analyzer.hop_size = 0.5
        results_low_conf, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
        print(f"    {len(results_low_conf)}개 감지")
        
        # 4. 높은 신뢰도 임계값
        print("\n[4] 높은 신뢰도 임계값 (0.9)")
        analyzer.confidence_threshold = 0.9
        results_high_conf, duration = analyzer.analyze_continuous_audio(audio_path, method='sliding_window')
        print(f"    {len(results_high_conf)}개 감지")
        
        print(f"\n[결과] 파라미터별 결과 비교:")
        print(f"    작은 윈도우 (1초): {len(results_small)}개")
        print(f"    큰 윈도우 (3초): {len(results_large)}개")
        print(f"    낮은 신뢰도 (0.5): {len(results_low_conf)}개")
        print(f"    높은 신뢰도 (0.9): {len(results_high_conf)}개")
        
    except Exception as e:
        print(f"[실패] 파라미터 테스트 실패: {e}")

if __name__ == "__main__":
    # 기본 테스트
    test_with_sample_audio()
    
    # 파라미터 조정 테스트
    test_parameter_adjustment()
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("결과 파일들은 output 폴더에서 확인할 수 있습니다.")
    print("="*60)
