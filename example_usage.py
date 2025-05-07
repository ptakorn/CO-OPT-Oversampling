# example_usage.py

from coopt import run_cog_v2
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ================================
# Example 1: Using F1-score
# ================================
final_score_f1, synthetic_f1 = run_cog_v2(
    file_path='your_dataset.csv',  # replace with actual path
    n_clusters=5,
    target_ir=0.6,
    objective_func=f1_score,
    patience=3
)
print("=== Example 1 ===")
print(f"Final F1-Score: {final_score_f1:.4f}")
print(f"Synthetic Samples: {synthetic_f1}\n")

# ================================
# Example 2: Using custom HPN and Random Forest
# ================================

def custom_hpn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP, FP = cm[1,1], cm[0,1]
    FN, TN = cm[1,0], cm[0,0]
    precision = TP / (TP + FP) if (TP + FP) else 0
    npv = TN / (TN + FN) if (TN + FN) else 0
    return (2 * precision * npv) / (precision + npv) if (precision + npv) else 0

final_score_hpn, synthetic_hpn = run_cog_v2(
    file_path='your_dataset.csv',  # replace with actual path
    n_clusters=7,
    target_ir=0.5,
    objective_func=custom_hpn,
    patience=2,
    base_classifier=RandomForestClassifier(random_state=42)
)
print("=== Example 2 ===")
print(f"Final HPN Score: {final_score_hpn:.4f}")
print(f"Synthetic Samples: {synthetic_hpn}")
