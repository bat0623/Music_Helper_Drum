import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# ▼▼▼ 1. 데이터 로딩 및 전처리 ▼▼▼
# =====================================================================

class DrumDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.labels = ['bass_drum', 'crash_cymbal', 'high_tom', 'hihat_closed', 
                      'hihat_open', 'low_tom', 'mid_tom', 'ride_cymbal', 'snare_drum']
        self.num_classes = len(self.labels)
        
    def load_data(self):
        """증강된 데이터 로딩"""
        print("데이터 로딩 중...")
        
        X = np.load(os.path.join(self.data_path, "X_data.npy"))
        y = np.load(os.path.join(self.data_path, "y_data.npy"))
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"클래스 수: {self.num_classes}")
        
        # 라벨을 원-핫 인코딩으로 변환
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        return X, y_categorical
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """데이터를 훈련/검증/테스트 세트로 분할"""
        print("데이터 분할 중...")
        
        # 먼저 훈련+검증 세트와 테스트 세트 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 훈련 세트와 검증 세트 분할
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"훈련 세트: {X_train.shape}")
        print(f"검증 세트: {X_val.shape}")
        print(f"테스트 세트: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# =====================================================================
# ▼▼▼ 2. CRNN 모델 아키텍처 정의 ▼▼▼
# =====================================================================

class CRNNDrumClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """CRNN 모델 구축"""
        print("CRNN 모델 구축 중...")
        
        inputs = keras.Input(shape=self.input_shape)
        
        # CNN 부분 - 특징 추출
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # CNN 출력을 RNN 입력 형태로 변환
        # (batch, height, width, channels) -> (batch, time_steps, features)
        shape = x.shape
        x = layers.Reshape((shape[2], shape[1] * shape[3]))(x)  # (batch, width, height*channels)
        
        # RNN 부분 - 시간적 패턴 학습
        x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(64, dropout=0.3)(x)
        
        # 완전연결층
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # 출력층
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # 모델 컴파일
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("모델 구축 완료!")
        return self.model
    
    def get_model_summary(self):
        """모델 구조 출력"""
        if self.model is None:
            print("모델이 구축되지 않았습니다.")
            return
        
        print("\n" + "="*60)
        print("CRNN 모델 구조")
        print("="*60)
        self.model.summary()

# =====================================================================
# ▼▼▼ 3. 모델 훈련 클래스 ▼▼▼
# =====================================================================

class ModelTrainer:
    def __init__(self, model, save_path="models"):
        self.model = model
        self.save_path = save_path
        self.history = None
        
        # 저장 경로 생성
        os.makedirs(save_path, exist_ok=True)
        
    def setup_callbacks(self):
        """훈련 콜백 설정"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.save_path, 'best_crnn_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """모델 훈련"""
        print("모델 훈련 시작...")
        
        callbacks = self.setup_callbacks()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("모델 훈련 완료!")
        return self.history
    
    def plot_training_history(self):
        """훈련 과정 시각화"""
        if self.history is None:
            print("훈련 기록이 없습니다.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 정확도 그래프
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # 손실 그래프
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

# =====================================================================
# ▼▼▼ 4. 모델 평가 클래스 ▼▼▼
# =====================================================================

class ModelEvaluator:
    def __init__(self, model, labels, save_path="models"):
        self.model = model
        self.labels = labels
        self.save_path = save_path
        
    def evaluate(self, X_test, y_test):
        """모델 평가"""
        print("모델 평가 중...")
        
        # 예측 수행
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # 정확도 계산
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"테스트 정확도: {test_accuracy:.4f}")
        print(f"테스트 손실: {test_loss:.4f}")
        
        # 분류 보고서
        print("\n분류 보고서:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=self.labels))
        
        # 혼동 행렬 시각화
        self.plot_confusion_matrix(y_true_classes, y_pred_classes)
        
        return test_accuracy, test_loss
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """혼동 행렬 시각화"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.labels, yticklabels=self.labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()

# =====================================================================
# ▼▼▼ 5. 메인 실행 함수 ▼▼▼
# =====================================================================

def main():
    # 프로젝트 루트 경로 자동 감지
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 경로 설정
    DATA_PATH = os.path.join(project_root, "data_augmented")
    MODEL_SAVE_PATH = script_dir  # models 폴더
    
    print("="*60)
    print("CRNN 드럼 분류 모델 훈련 시작")
    print("="*60)
    
    # 1. 데이터 로딩
    data_loader = DrumDataLoader(DATA_PATH)
    X, y = data_loader.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
    
    # 2. 모델 구축
    input_shape = X_train.shape[1:]  # (height, width, channels)
    crnn_model = CRNNDrumClassifier(input_shape, data_loader.num_classes)
    model = crnn_model.build_model()
    crnn_model.get_model_summary()
    
    # 3. 모델 훈련
    trainer = ModelTrainer(model, MODEL_SAVE_PATH)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    trainer.plot_training_history()
    
    # 4. 모델 평가
    evaluator = ModelEvaluator(model, data_loader.labels, MODEL_SAVE_PATH)
    test_accuracy, test_loss = evaluator.evaluate(X_test, y_test)
    
    # 5. 최종 모델 저장
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'final_crnn_drum_model.h5')
    model.save(final_model_path)
    print(f"\n최종 모델이 저장되었습니다: {final_model_path}")
    
    print("\n" + "="*60)
    print("훈련 완료!")
    print(f"최종 테스트 정확도: {test_accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
