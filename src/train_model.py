"""
train_model.py
--------------
Train and compare ML classifiers for Pre-Impact Rider State Classification.

Models compared:
    - Decision Tree
    - Random Forest
    - SVM
    - KNN
    - XGBoost

Best model is saved to models/random_forest.pkl (or best_model.pkl)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed, skipping XGBClassifier.")

LABEL_NAMES = {0: 'Normal', 1: 'Pothole', 2: 'SuddenBrake', 3: 'Crash'}
COLORS      = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']


# ─────────────────────────────────────────────
#  BUILD MODELS
# ─────────────────────────────────────────────

def get_models() -> dict:
    """Return dict of model_name → sklearn pipeline."""
    models = {
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=15,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
                probability=True
            ))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                n_jobs=-1
            ))
        ]),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )

    return models


# ─────────────────────────────────────────────
#  EVALUATE ONE MODEL
# ─────────────────────────────────────────────

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str
) -> dict:
    """Train + evaluate a single model. Returns metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average='macro')
    f1_wt   = f1_score(y_test, y_pred, average='weighted')
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  F1 (macro)        : {f1_mac:.4f}")
    print(f"  F1 (weighted)     : {f1_wt:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values()))}")

    return {
        "model_name" : model_name,
        "model"      : model,
        "accuracy"   : acc,
        "f1_macro"   : f1_mac,
        "f1_weighted": f1_wt,
        "confusion"  : cm,
        "y_test"     : y_test,
        "y_pred"     : y_pred,
    }


# ─────────────────────────────────────────────
#  COMPARE ALL MODELS — BAR CHART
# ─────────────────────────────────────────────

def plot_comparison(results: list, save_path: str = None):
    """Bar chart comparing all models on Accuracy and F1."""
    names = [r['model_name'] for r in results]
    accs  = [r['accuracy']   for r in results]
    f1s   = [r['f1_macro']   for r in results]

    x    = np.arange(len(names))
    w    = 0.35
    grad = ['#4facfe', '#00f2fe']

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a2e')

    bars1 = ax.bar(x - w/2, accs, w, label='Accuracy',  color='#4facfe', alpha=0.85, edgecolor='#0d0d0d')
    bars2 = ax.bar(x + w/2, f1s,  w, label='F1 (macro)', color='#f093fb', alpha=0.85, edgecolor='#0d0d0d')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', color='white', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, color='#e0e0e0', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', color='#aaaaaa')
    ax.set_title('Model Comparison — Pre-Impact Rider State Classification', color='white', fontsize=12)
    ax.tick_params(colors='#aaaaaa')
    ax.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white')
    ax.grid(True, axis='y', color='#222244', linewidth=0.5)
    for spine in ax.spines.values(): spine.set_edgecolor('#333355')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.show()


# ─────────────────────────────────────────────
#  CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────

def plot_confusion_matrix(result: dict, save_path: str = None):
    """Plot a styled confusion matrix for one model."""
    cm   = result['confusion']
    name = result['model_name']
    lbls = list(LABEL_NAMES.values())

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a2e')

    # Normalize for colors
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1)

    for i in range(len(lbls)):
        for j in range(len(lbls)):
            txt_color = 'black' if cm_norm[i, j] > 0.6 else 'white'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})',
                    ha='center', va='center', color=txt_color, fontsize=9)

    ax.set_xticks(range(len(lbls)))
    ax.set_yticks(range(len(lbls)))
    ax.set_xticklabels(lbls, color='#e0e0e0', rotation=15)
    ax.set_yticklabels(lbls, color='#e0e0e0')
    ax.set_xlabel('Predicted',  color='#aaaaaa')
    ax.set_ylabel('Actual',     color='#aaaaaa')
    ax.set_title(f'Confusion Matrix — {name}', color='white', fontsize=12)

    plt.colorbar(im, ax=ax, label='Normalized')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.show()


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE (RF / XGB)
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, top_n: int = 20, title: str = ""):
    """Bar chart of top feature importances."""
    # Unwrap pipeline if needed
    clf = model
    if hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf', model)

    if not hasattr(clf, 'feature_importances_'):
        print("Model doesn't support feature_importances_. Skipping.")
        return

    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_feats   = [feature_names[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#1a1a2e')

    bars = ax.barh(range(top_n), top_vals[::-1], color='#4facfe', alpha=0.85, edgecolor='#0d0d0d')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats[::-1], color='#e0e0e0', fontsize=8)
    ax.set_xlabel('Importance', color='#aaaaaa')
    ax.set_title(f'Top {top_n} Feature Importances — {title}', color='white', fontsize=12)
    ax.tick_params(colors='#aaaaaa')
    ax.grid(True, axis='x', color='#222244', linewidth=0.5)
    for spine in ax.spines.values(): spine.set_edgecolor('#333355')

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  SAVE BEST MODEL
# ─────────────────────────────────────────────

def save_best_model(results: list, feature_names: list, save_dir: str = "../models"):
    """Pick the best model by F1 macro and save it."""
    os.makedirs(save_dir, exist_ok=True)
    best = max(results, key=lambda r: r['f1_macro'])
    model_path = os.path.join(save_dir, "best_model.pkl")
    meta_path  = os.path.join(save_dir, "model_meta.pkl")

    joblib.dump(best['model'], model_path)
    joblib.dump({
        "model_name"   : best['model_name'],
        "accuracy"     : best['accuracy'],
        "f1_macro"     : best['f1_macro'],
        "feature_names": feature_names,
        "label_map"    : LABEL_NAMES
    }, meta_path)

    print(f"\n✅ Best model : {best['model_name']}")
    print(f"   Accuracy   : {best['accuracy']:.4f}")
    print(f"   F1 (macro) : {best['f1_macro']:.4f}")
    print(f"   Saved to   : {model_path}")
    print(f"   Meta saved : {meta_path}")

    # Also save RF specifically (for compatibility)
    rf_result = next((r for r in results if 'Random Forest' in r['model_name']), None)
    if rf_result:
        rf_path = os.path.join(save_dir, "random_forest.pkl")
        joblib.dump(rf_result['model'], rf_path)
        print(f"   RF saved   : {rf_path}")

    return best


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.feature_engineering import apply_sliding_window, split_dataset

    raw_path  = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "helmet_imu_raw.csv")
    feat_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    print("Loading raw data...")
    df = pd.read_csv(raw_path)

    print("Extracting features...")
    feature_df = apply_sliding_window(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(feature_df)

    feature_cols = [c for c in feature_df.columns if c not in ('label', 'session_id')]

    print("\nTraining models...")
    models  = get_models()
    results = []
    for name, model in models.items():
        res = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(res)

    plot_comparison(results)
    best = save_best_model(results, feature_cols, model_dir)
    plot_confusion_matrix(best)
    plot_feature_importance(best['model'], feature_cols, title=best['model_name'])
