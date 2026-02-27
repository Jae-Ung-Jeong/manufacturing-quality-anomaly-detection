import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

def load_models():
    """저장된 모델 불러오기"""
    xgb = joblib.load('models/xgb_model.pkl')
    iso_forest = joblib.load('models/iso_forest.pkl')
    scaler_iso = joblib.load('models/scaler_iso.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return xgb, iso_forest, scaler_iso, feature_cols

def preprocess_new_data(data: pd.DataFrame, feature_cols: list):
    """새 데이터 전처리"""
    # 결측치 80% 이상 피처 삭제
    missing_ratio = data.isnull().sum() / len(data) * 100
    cols_to_drop = missing_ratio[missing_ratio > 80].index
    data = data.drop(columns=cols_to_drop)

    # 중앙값 대체
    data = data.fillna(data.median())

    # 분산 0 피처 삭제
    variance = data.var()
    cols_zero_var = variance[variance == 0].index
    data = data.drop(columns=cols_zero_var)

    # 학습 때 사용한 피처만 선택 (컬럼 맞추기)
    data = data[feature_cols]

    return data

def predict(data: pd.DataFrame):
    """
    새 데이터 입력 → 전처리 → 예측 → 결과 반환
    """
    # 모델 로드
    xgb, iso_forest, scaler_iso, feature_cols = load_models()

    # 전처리
    data_processed = preprocess_new_data(data, feature_cols)

    # XGBoost 예측 확률
    xgb_prob = xgb.predict_proba(data_processed)[:, 1]

    # Isolation Forest 점수
    iso_scores = iso_forest.decision_function(data_processed)
    iso_scores_normalized = scaler_iso.transform(iso_scores.reshape(-1, 1)).flatten()
    iso_scores_inverted = 1 - iso_scores_normalized

    # 앙상블
    ensemble_prob = (xgb_prob + iso_scores_inverted) / 2

    # 임계값 0.25 적용
    predictions = (ensemble_prob >= 0.25).astype(int)

    # 결과 데이터프레임
    results = pd.DataFrame({
        'xgb_prob': xgb_prob.round(4),
        'iso_score': iso_scores_inverted.round(4),
        'ensemble_prob': ensemble_prob.round(4),
        'prediction': predictions,
        'result': ['Defect' if p == 1 else 'Normal' for p in predictions]
    })

    return results

if __name__ == "__main__":
    # 테스트: 전처리된 데이터로 예측
    data = pd.read_csv('data/secom_preprocessed.csv')
    X = data.drop(columns=['label'])
    
    results = predict(X)
    
    # 결과 저장
    results.to_csv('reports/prediction_results.csv', index=False)
    
    print("=== 예측 완료 ===")
    print(f"전체 샘플: {len(results)}")
    print(f"정상 판정: {(results['prediction'] == 0).sum()}")
    print(f"불량 판정: {(results['prediction'] == 1).sum()}")
    print(f"\n결과 저장: reports/prediction_results.csv")