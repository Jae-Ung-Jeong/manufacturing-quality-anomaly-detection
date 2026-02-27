import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess(data: pd.DataFrame, scaler=None, fit=False):
    """
    새 데이터 전처리 함수
    - 결측치 80% 이상 피처 삭제
    - 중앙값 대체
    - 분산 0 피처 삭제
    - 정규화
    """
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

    # 정규화
    if fit:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        joblib.dump(scaler, 'models/scaler.pkl')
    else:
        data_scaled = scaler.transform(data)

    return pd.DataFrame(data_scaled, columns=data.columns), scaler

print("pipeline.py 로드 완료")