import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

class AccountValidator:
    def __init__(self, max_length=20):
        self.max_length = max_length
        self.num_chars = 10  # 0-9만 사용
        self.bank_encoder = LabelEncoder()
        self.model = None
        
    def is_valid_account_format(self, account_number):
        """계좌번호가 유효한 형식인지 검사합니다."""
        # 숫자를 문자열로 변환
        account_number = str(account_number)
        
        # 숫자만 추출
        digits = ''.join(filter(str.isdigit, account_number))
        
        # 일반적인 길이 검사 (10-16자리)
        if not (10 <= len(digits) <= 16):
            return False
            
        return True
        
    def preprocess_account(self, account_number):
        """계좌번호를 전처리하여 원-핫 인코딩으로 변환합니다."""
        # 숫자를 문자열로 변환
        account_number = str(account_number)
        
        # 숫자만 추출
        digits = ''.join(filter(str.isdigit, account_number))
        
        if not self.is_valid_account_format(account_number):
            raise ValueError(f"유효하지 않은 계좌번호 형식: {account_number}")
        
        # 길이 정규화
        if len(digits) < self.max_length:
            digits = digits.zfill(self.max_length)
        else:
            digits = digits[:self.max_length]
        
        # 원-핫 인코딩 (0-9만 사용)
        encoded = np.zeros((self.max_length, self.num_chars))
        for i, digit in enumerate(digits):
            encoded[i, int(digit)] = 1
        
        return encoded
    
    def create_model(self, num_banks):
        """계좌번호 검증과 은행 분류를 위한 모델을 생성합니다."""
        # 입력층
        input_layer = layers.Input(shape=(self.max_length, self.num_chars))
        
        # LSTM 층들
        x = layers.LSTM(128, return_sequences=True)(input_layer)
        x = layers.LSTM(64)(x)
        
        # 공통 특성 추출
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # 계좌번호 유효성 검증 출력
        validity_output = layers.Dense(1, activation='sigmoid', name='validity')(x)
        
        # 은행 분류 출력
        bank_output = layers.Dense(num_banks, activation='softmax', name='bank')(x)
        
        # 모델 생성
        model = models.Model(
            inputs=input_layer,
            outputs=[validity_output, bank_output]
        )
        
        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss={
                'validity': 'binary_crossentropy',
                'bank': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'validity': 1.0,
                'bank': 1.0
            },
            metrics={
                'validity': 'accuracy',
                'bank': 'accuracy'
            }
        )
        
        return model
    
    def load_data(self, tsv_path):
        """TSV 파일에서 데이터를 로드하고 전처리합니다."""
        # TSV 파일 읽기
        df = pd.read_csv(tsv_path, sep='\t')
        
        # 유효하지 않은 계좌번호 제거
        valid_mask = df['account_number'].apply(self.is_valid_account_format)
        df = df[valid_mask].copy()
        
        # 은행 코드 검증
        bank_code_mask = df.apply(
            lambda row: self.validate_bank_code(row['account_number'], row['bank_name']),
            axis=1
        )
        df = df[bank_code_mask].copy()
        
        if len(df) == 0:
            raise ValueError("유효한 계좌번호가 없습니다.")
        
        # 계좌번호 전처리
        X = np.array([self.preprocess_account(acc) for acc in df['account_number']])
        
        # 은행명 인코딩
        y_bank = self.bank_encoder.fit_transform(df['bank_name'])
        
        # 계좌번호 유효성 레이블
        y_validity = np.ones(len(df))
        
        return X, y_validity, y_bank
    
    def train(self, tsv_path, epochs=50, batch_size=32):
        """모델을 학습시킵니다."""
        # 데이터 로드
        X, y_validity, y_bank = self.load_data(tsv_path)
        
        # 데이터 분할
        X_train, X_val, y_validity_train, y_validity_val, y_bank_train, y_bank_val = train_test_split(
            X, y_validity, y_bank, test_size=0.2, random_state=42
        )
        
        # 모델 생성
        self.model = self.create_model(len(self.bank_encoder.classes_))
        
        # 모델 학습
        history = self.model.fit(
            X_train,
            {
                'validity': y_validity_train,
                'bank': y_bank_train
            },
            validation_data=(
                X_val,
                {
                    'validity': y_validity_val,
                    'bank': y_bank_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def predict(self, account_number):
        """계좌번호의 유효성과 은행을 예측합니다."""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 계좌번호 전처리
        X = self.preprocess_account(account_number)
        X = np.expand_dims(X, axis=0)
        
        # 예측
        validity_score, bank_probs = self.model.predict(X)
        
        # 은행명 디코딩
        bank_idx = np.argmax(bank_probs[0])
        bank_name = self.bank_encoder.inverse_transform([bank_idx])[0]
        
        # 은행 코드 검증
        bank_code = self.get_bank_code(account_number)
        bank_code_valid = self.validate_bank_code(account_number, bank_name)
        
        # 신뢰도 레벨 결정
        confidence_level = "낮음"
        if validity_score[0][0] >= 0.95:
            confidence_level = "매우 높음"
        elif validity_score[0][0] >= 0.90:
            confidence_level = "높음"
        elif validity_score[0][0] >= 0.85:
            confidence_level = "중간"
        
        return {
            'validity_score': float(validity_score[0][0]),
            'bank_name': bank_name,
            'bank_probability': float(bank_probs[0][bank_idx]),
            'confidence_level': confidence_level,
            'is_valid': validity_score[0][0] >= 0.90 and bank_code_valid,
            'bank_code': bank_code,
            'bank_code_valid': bank_code_valid
        }
    
    def save_model(self, path):
        """모델을 저장합니다."""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 모델 저장
        self.model.save(f"{path}/model.h5")
        
        # 은행 인코더 저장
        np.save(f"{path}/bank_encoder.npy", self.bank_encoder.classes_)
    
    def load_model(self, path):
        """저장된 모델을 로드합니다."""
        # 모델 로드
        self.model = models.load_model(f"{path}/model.h5")
        
        # 은행 인코더 로드
        self.bank_encoder.classes_ = np.load(f"{path}/bank_encoder.npy")
    
    def get_bank_code(self, account_number):
        """계좌번호의 은행 코드를 반환합니다."""
        digits = ''.join(filter(str.isdigit, str(account_number)))
        if len(digits) >= 3:
            return digits[:3]
        return None

    def validate_bank_code(self, account_number, bank_name):
        """계좌번호의 은행 코드가 은행명과 일치하는지 검증합니다."""
        bank_codes = {
            '신한은행': '110',
            '국민은행': '123',
            '기업은행': '123',
            '농협은행': '123',
            '하나은행': '123',
            '우리은행': '123',
            'SC제일은행': '123',
            '시티은행': '123',
            '카카오뱅크': '333',
            '케이뱅크': '123'
        }
        
        account_code = self.get_bank_code(account_number)
        expected_code = bank_codes.get(bank_name)
        
        return account_code == expected_code 