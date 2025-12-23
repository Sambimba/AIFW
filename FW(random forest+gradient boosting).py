# -*- coding: utf-8 -*-
!pip install -q pandas numpy scikit-learn imbalanced-learn matplotlib joblib kagglehub seaborn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import joblib
import kagglehub

OUT_DIR = "firewall_outputs_unsw"
MODEL_DIR = f"{OUT_DIR}/models"
FIG_DIR = f"{OUT_DIR}/figures"
BLACKLIST_PATH = f"{OUT_DIR}/blacklist.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def load_unsw_nb15(
    max_samples=80000,
    attack_ratio=0.5
):
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    print(f"Dataset path: {path}")

    train_file = os.path.join(path, "UNSW_NB15_training-set.csv")
    test_file = os.path.join(path, "UNSW_NB15_testing-set.csv")

    print(f"Training file: {train_file}")
    print(f"Testing file: {test_file}")


    train_df = pd.read_csv(train_file, nrows=max_samples)

    test_df = pd.read_csv(test_file, nrows=int(max_samples * 0.3))

    df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"\nDataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    df.columns = df.columns.str.strip().str.lower()

    # Create binary label (0 = normal, 1 = attack)
    if 'label' in df.columns:
        df['y'] = df['label'].astype(int)
        print("Using 'label' column for binary classification")
    else:
        raise ValueError("No 'label' column found in dataset")

    print(f"\nClass distribution:")
    class_dist = df['y'].value_counts()
    print(class_dist)
    print(f"Attack ratio: {df['y'].mean():.3f}")

    if attack_ratio < 1.0:
        normal_indices = df[df['y'] == 0].index
        attack_indices = df[df['y'] == 1].index

        desired_normal = len(normal_indices)
        desired_attacks = int(desired_normal * attack_ratio / (1 - attack_ratio))

        if len(attack_indices) > desired_attacks:
            attack_indices = np.random.choice(attack_indices, desired_attacks, replace=False)

        selected_indices = np.concatenate([normal_indices, attack_indices])
        np.random.shuffle(selected_indices)

        df = df.loc[selected_indices].reset_index(drop=True)

        print(f"\nAfter balancing to attack_ratio={attack_ratio}:")
        print(f"Normal: {len(df[df['y'] == 0])}")
        print(f"Attack: {len(df[df['y'] == 1])}")
        print(f"New attack ratio: {df['y'].mean():.3f}")

    columns_to_drop = ['id', 'label', 'attack_cat']

    categorical_cols = ['proto', 'service', 'state']
    columns_to_drop.extend(categorical_cols)

    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    print(f"\nDropping columns: {columns_to_drop}")
    df_clean = df.drop(columns=columns_to_drop)

    X = df_clean.drop(columns='y')
    y = df['y'].values

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
            print(f"Filled NaN in column '{col}' with median")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, f"{MODEL_DIR}/unsw_scaler.pkl")

    return X_scaled, y, df

class BlacklistManager:
    def __init__(self):
        self.records = []

    def add(self, df_rows, predictions, proba=None, threshold=0.7):
        if proba is not None:
            high_conf_attacks = proba[:, 1] > threshold
            attacks = df_rows[high_conf_attacks]
        else:
            attacks = df_rows[predictions == 1]

        if len(attacks) > 0:
            self.records.append(attacks)
            print(f"Added {len(attacks)} attacks to blacklist")

    def save(self, path):
        if self.records:
            blacklist = pd.concat(self.records, ignore_index=True)

            blacklist['detection_timestamp'] = pd.Timestamp.now()
            blacklist['source'] = 'UNSW-NB15 IDS'

            blacklist.to_csv(path, index=False)
            print(f"\nFirewall blacklist size: {len(blacklist)}")
            print(f"Saved to: {path}")

            if 'attack_cat' in blacklist.columns:
                print("\nAttack categories in blacklist:")
                print(blacklist['attack_cat'].value_counts())
        else:
            print("Blacklist is empty")

def train_ids_models(X, y, df_raw):
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)),
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    df_tr = df_raw.iloc[idx_tr]
    df_te = df_raw.iloc[idx_te]

    print(f"\nTraining set: {Xtr.shape}, Attack ratio: {ytr.mean():.3f}")
    print(f"Test set: {Xte.shape}, Attack ratio: {yte.mean():.3f}")

    models = {
        "RandomForest_UNSW": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        "GradientBoosting_UNSW": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }

    results = []
    firewall = BlacklistManager()

    for name, clf in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print('='*60)

        print(f"Class distribution before SMOTE: Normal={sum(ytr==0)}, Attack={sum(ytr==1)}")

        if ytr.mean() < 0.5:  # Attacks are minority
            print("Applying SMOTE (attacks are minority class)")
            pipe = Pipeline([
                ('smote', SMOTE(random_state=42, sampling_strategy='minority')),
                ('clf', clf)
            ])
        else:
            print("Attacks are majority class, using RandomUnderSampler")
            pipe = Pipeline([
                ('undersample', RandomUnderSampler(random_state=42, sampling_strategy=0.5)),
                ('clf', clf)
            ])

        pipe.fit(Xtr, ytr)

        yp = pipe.predict(Xte)

        if hasattr(pipe, 'predict_proba'):
            yp_proba = pipe.predict_proba(Xte)
            auc_score = roc_auc_score(yte, yp_proba[:, 1])
        else:
            yp_proba = None
            auc_score = 0

        cm = confusion_matrix(yte, yp)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn + 1e-9)
            tnr = tn / (tn + fp + 1e-9)
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            fpr = 0
            tnr = 0

        precision = precision_score(yte, yp, zero_division=0)
        recall = recall_score(yte, yp, zero_division=0)
        f1 = f1_score(yte, yp, zero_division=0)
        acc = accuracy_score(yte, yp)

        print("\nConfusion Matrix:")
        print(cm)
        print(f"\nDetailed Metrics:")
        print(f"True Positives (TP):  {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Negatives (TN):  {tn}")
        print(f"Precision:           {precision:.4f}")
        print(f"Recall/Sensitivity:  {recall:.4f}")
        print(f"Specificity:         {tnr:.4f}")
        print(f"FPR:                 {fpr:.4f}")
        print(f"F1-score:            {f1:.4f}")
        print(f"Accuracy:            {acc:.4f}")
        if auc_score > 0:
            print(f"AUC-ROC:             {auc_score:.4f}")

        print("\nClassification Report:")
        print(classification_report(yte, yp, target_names=['Normal', 'Attack']))

        results.append([name, precision, recall, fpr, f1, acc, auc_score, tp, fp, fn, tn])

        # Add to blacklist (using high-confidence predictions)
        if yp_proba is not None:
            firewall.add(df_te, yp, yp_proba, threshold=0.8)
        else:
            firewall.add(df_te, yp)

        model_path = f"{MODEL_DIR}/{name}.pkl"
        joblib.dump(pipe, model_path)
        print(f"\nModel saved to: {model_path}")

        plot_confusion_matrix(cm, name)

    firewall.save(BLACKLIST_PATH)

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Precision", "Recall", "FPR", "F1-score", "Accuracy",
                 "AUC-ROC", "TP", "FP", "FN", "TN"]
    )

    return results_df

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal', 'Predicted Attack'],
                yticklabels=['Actual Normal', 'Actual Attack'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    fig_path = f"{FIG_DIR}/cm_{model_name}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to: {fig_path}")

def plot_model_comparison(results_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    models = results_df['Model'].values

    ax1 = axes[0, 0]
    ax1.plot(results_df['Recall'], results_df['Precision'], 'bo-', linewidth=2, markersize=10)
    for i, model in enumerate(models):
        ax1.annotate(model, (results_df['Recall'][i], results_df['Precision'][i]),
                    xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Trade-off')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    x = np.arange(len(models))
    width = 0.35
    ax2.bar(x - width/2, results_df['F1-score'], width, label='F1-score', alpha=0.8)
    ax2.bar(x + width/2, results_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Score')
    ax2.set_title('F1-score and Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    bars = ax3.bar(models, results_df['FPR'], color='red', alpha=0.7)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('False Positive Rate')
    ax3.set_title('False Positive Rate Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    ax4 = axes[1, 0]
    metrics_labels = ['TP', 'FP', 'FN', 'TN']
    for i, model in enumerate(models):
        metrics = [results_df['TP'][i], results_df['FP'][i],
                  results_df['FN'][i], results_df['TN'][i]]
        positions = [i*5 + j for j in range(4)]
        ax4.bar(positions, metrics, label=model if i==0 else "", alpha=0.7)

    ax4.set_ylabel('Count')
    ax4.set_title('Confusion Matrix Elements')
    ax4.set_xticks([i*5 + 1.5 for i in range(len(models))])
    ax4.set_xticklabels(models, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)


    ax5 = axes[1, 1]
    if 'AUC-ROC' in results_df.columns:
        ax5.bar(models, results_df['AUC-ROC'], color='green', alpha=0.7)
        ax5.set_xlabel('Model')
        ax5.set_ylabel('AUC-ROC Score')
        ax5.set_title('AUC-ROC Comparison')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)

        for i, (model, auc) in enumerate(zip(models, results_df['AUC-ROC'])):
            ax5.text(i, auc + 0.01, f'{auc:.3f}', ha='center')


    ax6 = axes[1, 2]
    metrics_to_plot = ['Precision', 'Recall', 'F1-score', 'Accuracy']
    if 'AUC-ROC' in results_df.columns:
        metrics_to_plot.append('AUC-ROC')

    for i, model in enumerate(models):
        values = [results_df[metric][i] for metric in metrics_to_plot]
        angles = np.linspace(0, 2*np.pi, len(values), endpoint=False).tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]

        ax6.plot(angles, values, 'o-', linewidth=2, label=model)
        ax6.fill(angles, values, alpha=0.1)

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics_to_plot)
    ax6.set_title('Model Performance Comparison')
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax6.grid(True)

    plt.tight_layout()


    fig_path = f"{FIG_DIR}/model_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nModel comparison plot saved to: {fig_path}")

def main():
    print("Starting UNSW-NB15 Intrusion Detection System")
    print("="*60)


    plt.style.use('seaborn-v0_8-darkgrid')

    X, y, df_raw = load_unsw_nb15(
        max_samples=80000,
        attack_ratio=0.5  # Balanced dataset
    )

    results = train_ids_models(X, y, df_raw)

    results_path = f"{OUT_DIR}/ids_results_unsw.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(results.to_string())

    plot_model_comparison(results)

    print("\n" + "="*60)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Output directory: {OUT_DIR}")
    print(f"Models saved in: {MODEL_DIR}")
    print(f"Figures saved in: {FIG_DIR}")
    print(f"Blacklist saved: {BLACKLIST_PATH}")

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print("\nBest model by metric:")
    print(f"Precision: {results.loc[results['Precision'].idxmax(), 'Model']} ({results['Precision'].max():.4f})")
    print(f"Recall:    {results.loc[results['Recall'].idxmax(), 'Model']} ({results['Recall'].max():.4f})")
    print(f"F1-score:  {results.loc[results['F1-score'].idxmax(), 'Model']} ({results['F1-score'].max():.4f})")
    print(f"Accuracy:  {results.loc[results['Accuracy'].idxmax(), 'Model']} ({results['Accuracy'].max():.4f})")

    if 'AUC-ROC' in results.columns:
        print(f"AUC-ROC:   {results.loc[results['AUC-ROC'].idxmax(), 'Model']} ({results['AUC-ROC'].max():.4f})")

if __name__ == "__main__":
    main()
