import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

# Default HPN scoring function with pos_label support
def hpn_score(y_true, y_pred, pos_label=1):
    y_true_bin = [1 if y == pos_label else 0 for y in y_true]
    y_pred_bin = [1 if y == pos_label else 0 for y in y_pred]
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    TP, FP = cm[1, 1], cm[0, 1]
    FN, TN = cm[1, 0], cm[0, 0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    return (2 * precision * npv) / (precision + npv) if (precision + npv) > 0 else 0

# Main CO-OPT function
def run_cog_v2(file_path, n_clusters, target_ir, objective_func=None,
               patience=3, base_classifier=None, minority_class=1):
    if base_classifier is None:
        base_classifier = DecisionTreeClassifier(random_state=42)
    if objective_func is None:
        objective_func = hpn_score  # Use default HPN if not provided

    # Load and split dataset
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Target'])
    y = df['Target']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Initial model evaluation
    model_init = clone(base_classifier)
    model_init.fit(X_train, y_train)
    best_score = objective_func(y_val, model_init.predict(X_val), pos_label=minority_class)

    # K-means clustering on training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_train)
    S = pd.DataFrame(X_train, columns=X.columns)
    S['Target'] = y_train.values
    S['Cluster'] = clusters

    resampling_counts = {i: 0 for i in range(n_clusters)}
    total_synthetic = 0

    for i in range(n_clusters):
        cluster_data = S[S['Cluster'] == i]
        counts = cluster_data['Target'].value_counts()
        minority_count = counts.get(minority_class, 0)
        if minority_count < 2:
            continue

        majority_class_label = counts.drop(index=minority_class).idxmax()
        majority_count = counts.get(majority_class_label, 0)
        if majority_count == 0:
            continue

        current_ir = minority_count / majority_count
        no_improve_count = 0

        while current_ir < target_ir:
            smote_ratio = min((target_ir - current_ir) / (1 - current_ir), 0.5)
            smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)

            try:
                X_res, y_res = smote.fit_resample(
                    cluster_data.drop(columns=['Target', 'Cluster']),
                    cluster_data['Target']
                )
            except ValueError:
                break

            added_instances = len(y_res) - len(cluster_data)

            temp_S = pd.concat([
                S.drop(S[S['Cluster'] == i].index),
                pd.concat([X_res, y_res], axis=1)
            ], ignore_index=True)
            temp_S['Cluster'] = -1

            model_tmp = clone(base_classifier)
            model_tmp.fit(temp_S.drop(columns=['Target', 'Cluster']), temp_S['Target'])
            score_tmp = objective_func(y_val, model_tmp.predict(X_val), pos_label=minority_class)

            if score_tmp > best_score:
                best_score = score_tmp
                S = temp_S
                counts_res = y_res.value_counts()
                minority_count = counts_res.get(minority_class, 0)
                majority_count = counts_res.drop(index=minority_class).max()
                current_ir = minority_count / majority_count
                resampling_counts[i] += added_instances
                total_synthetic += added_instances
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    break

    # Final model training and test evaluation
    X_final, y_final = S.drop(columns=['Target', 'Cluster']), S['Target']
    final_model = clone(base_classifier)
    final_model.fit(X_final, y_final)
    final_score = objective_func(y_test, final_model.predict(X_test), pos_label=minority_class)

    return final_score, total_synthetic
