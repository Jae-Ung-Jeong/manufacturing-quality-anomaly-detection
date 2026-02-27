import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
import joblib
import os

def generate_report(y_test, ensemble_prob, y_pred_final, save_path='reports/'):
    """
    분석 결과 시각화 리포트 생성
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle('SECOM Manufacturing Quality Analysis Report', 
                 fontsize=16, fontweight='bold', y=1.02)

    # 1. 클래스 분포
    ax1 = fig.add_subplot(gs[0, 0])
    labels_count = pd.Series(y_test).map({0: 'Normal', 1: 'Defect'}).value_counts()
    colors = ['steelblue', 'tomato']
    ax1.bar(labels_count.index, labels_count.values, color=colors, edgecolor='black')
    ax1.set_title('Class Distribution')
    ax1.set_ylabel('Count')
    for i, v in enumerate(labels_count.values):
        ax1.text(i, v + 1, str(v), ha='center', fontweight='bold')

    # 2. 예측 확률 분포
    ax2 = fig.add_subplot(gs[0, 1])
    normal_probs = ensemble_prob[y_test == 0]
    defect_probs = ensemble_prob[y_test == 1]
    ax2.hist(normal_probs, bins=30, alpha=0.6, color='steelblue', label='Normal')
    ax2.hist(defect_probs, bins=30, alpha=0.6, color='tomato', label='Defect')
    ax2.axvline(x=0.25, color='black', linestyle='--', label='Threshold=0.25')
    ax2.set_title('Prediction Probability Distribution')
    ax2.set_xlabel('Ensemble Probability')
    ax2.set_ylabel('Count')
    ax2.legend()

    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_test, ensemble_prob)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color='tomato', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax3.set_title('ROC Curve')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend()

    # 4. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Defect'],
                yticklabels=['Normal', 'Defect'], ax=ax4)
    ax4.set_title('Confusion Matrix (threshold=0.25)')
    ax4.set_ylabel('Actual')
    ax4.set_xlabel('Predicted')

    # 5. 임계값별 성능
    ax5 = fig.add_subplot(gs[1, 1])
    from sklearn.metrics import f1_score, recall_score, precision_score
    thresholds = np.arange(0.05, 0.6, 0.05)
    recalls, precisions, f1s = [], [], []
    for t in thresholds:
        y_pred_t = (ensemble_prob >= t).astype(int)
        recalls.append(recall_score(y_test, y_pred_t, pos_label=1, zero_division=0))
        precisions.append(precision_score(y_test, y_pred_t, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, pos_label=1, zero_division=0))
    ax5.plot(thresholds, recalls, 'o-', color='tomato', label='Recall')
    ax5.plot(thresholds, precisions, 's-', color='steelblue', label='Precision')
    ax5.plot(thresholds, f1s, '^-', color='green', label='F1')
    ax5.axvline(x=0.25, color='black', linestyle='--', label='Threshold=0.25')
    ax5.set_title('Threshold vs Metrics')
    ax5.set_xlabel('Threshold')
    ax5.set_ylabel('Score')
    ax5.legend()

    # 6. 성능 요약 텍스트
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary = f"""
    ■ 모델 구성
    - XGBoost + Isolation Forest 앙상블
    - SMOTE 클래스 불균형 해소
    
    ■ 데이터
    - 전체: 1,567개 샘플
    - 피처: 590 → 466개
    - 불량 비율: 6.6%
    
    ■ 최종 성능 (threshold=0.25)
    - ROC-AUC: {roc_auc:.3f}
    - Defect Recall: {recall_score(y_test, y_pred_final, pos_label=1, zero_division=0):.3f}
    - Defect F1: {f1_score(y_test, y_pred_final, pos_label=1, zero_division=0):.3f}
    
    ■ 핵심 인사이트
    - 불량 미탐지 > 과탐지 위험
    - Recall 최우선 지표 선택
    - 임계값 조정으로 트레이드오프 제어
    """
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax6.set_title('Summary')

    plt.tight_layout()
    plt.savefig(f'{save_path}final_report.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"리포트 저장 완료: {save_path}final_report.png")

if __name__ == "__main__":
    import pandas as pd
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # 데이터 로드
    data = pd.read_csv('data/secom_preprocessed.csv')
    X = data.drop(columns=['label'])
    y = data['label'].map({-1: 0, 1: 1})

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 모델 로드
    xgb = joblib.load('models/xgb_model.pkl')
    iso_forest = joblib.load('models/iso_forest.pkl')
    scaler_iso = joblib.load('models/scaler_iso.pkl')

    # 예측
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    iso_scores = iso_forest.decision_function(X_test)
    iso_scores_normalized = scaler_iso.transform(iso_scores.reshape(-1, 1)).flatten()
    iso_scores_inverted = 1 - iso_scores_normalized
    ensemble_prob = (xgb_prob + iso_scores_inverted) / 2
    y_pred_final = (ensemble_prob >= 0.25).astype(int)

    generate_report(y_test.values, ensemble_prob, y_pred_final)