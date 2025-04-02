import os
from model import AccountValidator

def main():
    # 모델 인스턴스 생성
    validator = AccountValidator()
    
    # 학습 데이터 경로
    tsv_path = "data/accounts.tsv"
    
    # 모델 저장 경로
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)
    
    # 모델 학습
    print("모델 학습 시작...")
    history = validator.train(
        tsv_path=tsv_path,
        epochs=50,
        batch_size=32
    )
    
    # 모델 저장
    print("모델 저장 중...")
    validator.save_model(model_path)
    
    # 테스트 예측
    test_accounts = [
        "110123456789",  # 신한은행
        "123456789012",  # 국민은행
        "123456789015",  # 기업은행
    ]
    
    print("\n테스트 예측:")
    for account in test_accounts:
        result = validator.predict(account)
        print(f"\n계좌번호: {account}")
        print(f"유효성 점수: {result['validity_score']:.4f}")
        print(f"예측 은행: {result['bank_name']}")
        print(f"은행 예측 확률: {result['bank_probability']:.4f}")

if __name__ == "__main__":
    main() 