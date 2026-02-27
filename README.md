# 제조 공정 데이터 기반 AI 이상 탐지 파이프라인
> 반도체 제조 공정 센서 데이터에서 불량을 조기 탐지하는 end-to-end AI 파이프라인

## 📌 프로젝트 개요
실제 반도체 제조 현장(SECOM)의 센서 데이터를 활용해 불량을 사전에 탐지하는 
AI 모델과 자동화 파이프라인을 구축했습니다.

단순 모델 구현에 그치지 않고, 새 데이터가 입력되면 전처리 → 예측 → 
결과 저장까지 자동으로 실행되는 체계를 만들었습니다.

## 🔍 문제 정의
- 제조 공정에서 불량 샘플 비율이 **6.6%** 에 불과한 극심한 클래스 불균형
- 590개 센서 피처 중 대부분에 결측치 존재
- 단순 정확도(Accuracy)로는 불량 탐지 불가 → **Recall, F1을 핵심 지표로 선택**

## 📊 데이터
- **출처:** UCI SECOM Dataset
- **샘플 수:** 1,567개
- **피처 수:** 590개 센서 피처
- **불량 비율:** 6.6% (정상 1,463개 / 불량 104개)

## ⚙️ 기술 스택
- **언어:** Python 3.11
- **라이브러리:** pandas, numpy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn
- **버전 관리:** Git / GitHub

## 🔄 파이프라인 구조
```
Raw Data (590 피처)
→ 전처리: 결측치 제거 + 중앙값 대체 + 분산0 제거 + 정규화
→ 466개 피처
→ SMOTE 클래스 불균형 해소
→ XGBoost + Isolation Forest 앙상블
→ 임계값 0.25 적용
→ 예측 결과 CSV 저장 + 리포트 자동 생성
```

## 🧠 모델링 전략

### 클래스 불균형 처리
- 불량 83개 → SMOTE로 1,170개 생성 (1:1 균형)
- 단순 오버샘플링이 아닌 보간 기반 샘플 생성

### 앙상블 구성
| 모델 | 역할 |
|------|------|
| XGBoost | 지도학습 기반 불량 분류 |
| Isolation Forest | 비지도학습 기반 이상 탐지 |
| 앙상블 | 두 모델 확률 평균으로 보완 |

### 임계값 최적화
- 제조 품질 특성상 **False Negative(불량 미탐지)가 False Positive보다 위험**
- Recall을 우선 지표로 설정, 임계값 0.25 선택

## 📈 최종 성능
| 지표 | 값 |
|------|-----|
| ROC-AUC | 0.676 |
| Defect Recall | 0.381 |
| Defect F1 | 0.219 |
| Normal Precision | 0.950 |

## 📁 프로젝트 구조
```
├── data/
│   ├── secom.data
│   ├── secom_labels.data
│   └── secom_preprocessed.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── pipeline.py
│   ├── predict.py
│   └── report.py
├── models/
│   ├── xgb_model.pkl
│   ├── iso_forest.pkl
│   └── scaler_iso.pkl
├── reports/
│   ├── final_report.png
│   └── confusion_matrix.png
├── docs/
│   └── log.md
└── README.md
```

## 🚀 실행 방법
```bash
# 예측 실행
python src/predict.py

# 리포트 생성
python src/report.py
```

## 💡 핵심 인사이트
1. **도메인 이해가 먼저** - 불량 미탐지가 과탐지보다 위험한 제조 특성 반영
2. **지표 선택의 중요성** - Accuracy 93%는 의미없음, Recall/F1이 핵심
3. **앙상블 효과** - 단일 모델 대비 불량 탐지율 향상
4. **임계값 조정** - 비즈니스 요구에 따라 Recall/Precision 트레이드오프 제어 가능

## 📝 작업 로그
[docs/log.md](docs/log.md) 참고