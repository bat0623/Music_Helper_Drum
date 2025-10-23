import sys
import os
import importlib.util
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# 숫자로 시작하는 모듈 import를 위한 특수 처리
spec = importlib.util.spec_from_file_location("score_analyzer", 
    os.path.join(os.path.dirname(__file__), '..', 'models', '4_score_analyzer.py'))
score_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_analyzer)
DrumScoreAnalyzer = score_analyzer.DrumScoreAnalyzer

def main():
    print("="*60)
    print("드럼 악보 분석 테스트")
    print("="*60)
    
    # 프로젝트 루트 경로 자동 감지
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 모델 경로
    MODEL_PATH = os.path.join(project_root, "models", "final_crnn_drum_model.h5")
    TEST_SOUND_DIR = os.path.join(project_root, "TestSound")
    OUTPUT_DIR = os.path.join(project_root, "output")
    
    # 분석기 초기화
    try:
        analyzer = DrumScoreAnalyzer(MODEL_PATH)
        print("분석기 초기화 성공")
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
    for i, f in enumerate(test_files, 1):
        print(f"{i}. {os.path.basename(f)}")
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
            try:
                analyzer.create_score_visualization(results, duration, test_audio, 
                                                   save_path=output_png)
                print(f"시각화 완료: {output_png}")
            except Exception as e:
                print(f"시각화 실패: {e}")
            
            # 결과 내보내기
            print(f"결과 내보내기...")
            try:
                analyzer.export_results(results, duration, output_txt)
                print(f"결과 파일 저장 완료: {output_txt}")
            except Exception as e:
                print(f"결과 내보내기 실패: {e}")
            
            print(f"\n{filename} 분석 완료")
            print("=" * 60)
            
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print(f"전체 분석 완료! 총 {len(test_files)}개 파일 처리됨")
    print(f"결과 파일 위치: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
