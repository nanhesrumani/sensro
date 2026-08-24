"""
generate_notebooks.py
Run once to create both Jupyter notebooks.
python ml/generate_notebooks.py
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src, "id": "md"+str(hash(src))[-6:]}

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None, "id": "c"+str(hash(src))[-6:]}

# ══════════════════════════════════════════════════════════════
# NOTEBOOK 1 — Exploration
# ══════════════════════════════════════════════════════════════
nb1_cells = [

md("# 01 — Data Exploration\nLoad raw sensor readings from SQLite and explore distributions, correlations, and class balance."),

code("""\
import sqlite3, os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

DB_PATH = os.path.join('..', 'road_data.db')   # adjust if needed
con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(\"\"\"
    SELECT r.*, t.name as trip_name
    FROM readings r
    JOIN trips t ON t.id = r.trip_id
    WHERE r.condition IN ('good','avg','bad')
\"\"\", con)
con.close()
print(f"Total rows: {len(df)}")
df.head()
"""),

md("## Class Distribution"),
code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = {'good':'#1a6','avg':'#ca0','bad':'#c33'}

counts = df['condition'].value_counts()
axes[0].bar(counts.index, counts.values,
            color=[colors[c] for c in counts.index], edgecolor='white')
axes[0].set_title('Class Distribution (count)')
axes[0].set_ylabel('Count')

pct = counts / counts.sum() * 100
axes[1].pie(pct, labels=pct.index, autopct='%1.1f%%',
            colors=[colors[c] for c in pct.index], startangle=90)
axes[1].set_title('Class Distribution (%)')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()
print(counts.to_string())
"""),

md("## Trips Overview"),
code("""\
trip_summary = df.groupby(['trip_id','trip_name','condition']).size().unstack(fill_value=0)
print(trip_summary)
"""),

md("## Sensor Feature Distributions by Class"),
code("""\
features = ['accel_magnitude','gyro_magnitude','speed','accel_z','accel_x','accel_y',
            'gyro_x','gyro_y','gyro_z','accuracy','altitude']
features = [f for f in features if f in df.columns]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, feat in enumerate(features):
    for cond, grp in df.groupby('condition'):
        axes[i].hist(grp[feat].dropna(), bins=40, alpha=0.6,
                     label=cond, color=colors[cond], density=True)
    axes[i].set_title(feat)
    axes[i].legend(fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Feature Distributions per Class', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150)
plt.show()
"""),

md("## Boxplots — Key Features vs Condition"),
code("""\
key = ['accel_magnitude','gyro_magnitude','speed','accel_z']
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, feat in zip(axes, key):
    order = ['good','avg','bad']
    data  = [df[df.condition==c][feat].dropna().values for c in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2))
    for patch, c in zip(bp['boxes'], order):
        patch.set_facecolor(colors[c])
    ax.set_title(feat)
plt.suptitle('Boxplots: Key Sensors vs Road Condition', fontsize=13)
plt.tight_layout()
plt.savefig('boxplots.png', dpi=150)
plt.show()
"""),

md("## Correlation Heatmap"),
code("""\
num_cols = df.select_dtypes(include=np.number).drop(
    columns=['id','trip_id','lat','lng'], errors='ignore')
corr = num_cols.corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', linewidths=0.5, annot_kws={'size':7})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
"""),

md("## Missing Values"),
code("""\
miss = df.isnull().sum().sort_values(ascending=False)
miss_pct = (miss / len(df) * 100).round(2)
summary = pd.DataFrame({'missing': miss, 'pct': miss_pct})
print(summary[summary.missing > 0])

plt.figure(figsize=(10,4))
miss_pct[miss_pct > 0].plot(kind='bar', color='#c55', edgecolor='white')
plt.title('Missing Values (%)')
plt.ylabel('%')
plt.tight_layout()
plt.savefig('missing_values.png', dpi=150)
plt.show()
"""),

md("## Speed vs Accel Magnitude (scatter by class)"),
code("""\
plt.figure(figsize=(10, 6))
for cond, grp in df.groupby('condition'):
    plt.scatter(grp['speed'].fillna(0)*3.6, grp['accel_magnitude'],
                alpha=0.5, label=cond, color=colors[cond], s=20)
plt.xlabel('Speed (km/h)')
plt.ylabel('Accel Magnitude')
plt.title('Speed vs Accel Magnitude by Condition')
plt.legend()
plt.tight_layout()
plt.savefig('speed_vs_accel.png', dpi=150)
plt.show()
"""),

md("## Per-Trip Timeline"),
code("""\
df['row_idx'] = df.groupby('trip_id').cumcount()
cmap = {'good':'#1a6','avg':'#ca0','bad':'#c33'}

trips = df['trip_id'].unique()[:6]  # show first 6 trips
fig, axes = plt.subplots(len(trips), 1, figsize=(14, 3*len(trips)), sharex=False)
if len(trips)==1: axes=[axes]

for ax, tid in zip(axes, trips):
    sub = df[df.trip_id==tid].copy()
    sub = sub.reset_index(drop=True)
    for cond, grp in sub.groupby('condition'):
        ax.scatter(grp.index, grp['accel_magnitude'],
                   color=cmap[cond], s=15, label=cond, alpha=0.8)
    ax.set_title(f"Trip #{tid} — accel_magnitude over time")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('trip_timelines.png', dpi=150)
plt.show()
"""),

md("## Summary Statistics per Class"),
code("""\
feat_cols = ['accel_magnitude','gyro_magnitude','speed','accel_z','accel_x','accel_y']
feat_cols = [f for f in feat_cols if f in df.columns]
stats = df.groupby('condition')[feat_cols].agg(['mean','std','min','max'])
print(stats.to_string())
stats.to_csv('class_stats.csv')
print('\\nSaved → class_stats.csv')
"""),
]

# ══════════════════════════════════════════════════════════════
# NOTEBOOK 2 — Train & Compare
# ══════════════════════════════════════════════════════════════
nb2_cells = [

md("# 02 — Model Training & Comparison\nTrain 6 classifiers on road condition sensor data, compare metrics, visualise results."),

code("""\
import sqlite3, os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
warnings.filterwarnings('ignore')

DB_PATH   = os.path.join('..', 'road_data.db')
MODEL_DIR = os.path.join('models')
os.makedirs(MODEL_DIR, exist_ok=True)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, confusion_matrix, classification_report,
                                     roc_auc_score, ConfusionMatrixDisplay)
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.neural_network  import MLPClassifier
from xgboost                 import XGBClassifier

COLORS = {'good':'#1a6','avg':'#ca0','bad':'#c33'}
print("Imports OK")
"""),

md("## Load & Engineer Features"),
code("""\
con = sqlite3.connect(DB_PATH)
df  = pd.read_sql_query(\"\"\"
    SELECT condition, speed, accel_x, accel_y, accel_z, accel_magnitude,
           gyro_x, gyro_y, gyro_z, gyro_magnitude,
           accuracy, altitude, heading
    FROM readings WHERE condition IN ('good','avg','bad')
\"\"\", con)
con.close()
print(f"Loaded {len(df)} rows | {df['condition'].value_counts().to_dict()}")

# ── Feature engineering ──────────────────────────────────────
df['accel_xy']     = np.sqrt(df.accel_x**2 + df.accel_y**2)
df['accel_xz']     = np.sqrt(df.accel_x**2 + df.accel_z**2)
df['accel_yz']     = np.sqrt(df.accel_y**2 + df.accel_z**2)
df['accel_norm_z'] = df.accel_z.abs() / (df.accel_magnitude + 1e-6)
df['gyro_xy']      = np.sqrt(df.gyro_x**2 + df.gyro_y**2)
df['gyro_total_turn'] = df.gyro_magnitude
df['speed_kmh']    = df.speed.fillna(0) * 3.6
df['speed_bucket'] = pd.cut(df.speed_kmh, bins=[-1,10,30,60,999], labels=[0,1,2,3]).astype(float)
df['rough_proxy']  = df.accel_magnitude / (df.speed_kmh + 1)
heading_rad        = np.deg2rad(df.heading.fillna(0))
df['heading_sin']  = np.sin(heading_rad)
df['heading_cos']  = np.cos(heading_rad)

FEATURE_COLS = [
    'speed','accel_x','accel_y','accel_z','accel_magnitude',
    'gyro_x','gyro_y','gyro_z','gyro_magnitude',
    'accuracy','altitude',
    'accel_xy','accel_xz','accel_yz','accel_norm_z',
    'gyro_xy','gyro_total_turn',
    'speed_kmh','speed_bucket','rough_proxy',
    'heading_sin','heading_cos'
]

df = df.dropna(subset=FEATURE_COLS, thresh=len(FEATURE_COLS)-3)
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

X   = df[FEATURE_COLS].values
le  = LabelEncoder()
y   = le.fit_transform(df['condition'].values)
print(f"Classes: {le.classes_} | Features: {len(FEATURE_COLS)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

dump(scaler, os.path.join(MODEL_DIR,'scaler.pkl'))
dump(le,     os.path.join(MODEL_DIR,'label_encoder.pkl'))
with open(os.path.join(MODEL_DIR,'feature_cols.json'),'w') as f:
    json.dump(FEATURE_COLS, f)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
"""),

md("## Define Models"),
code("""\
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=200, random_state=42), False),
    'XGBoost':             (XGBClassifier(n_estimators=200, use_label_encoder=False,
                                          eval_metric='mlogloss', random_state=42, verbosity=0), False),
    'SVM (RBF)':           (SVC(kernel='rbf', probability=True, random_state=42), True),
    'KNN':                 (KNeighborsClassifier(n_neighbors=7), True),
    'MLP':                 (MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42), True),
}
# (model_object, needs_scaling)
print("Models defined:", list(models.keys()))
"""),

md("## Train All Models + 5-Fold CV"),
code("""\
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, (model, needs_scale) in models.items():
    Xtr = X_train_s if needs_scale else X_train
    Xte = X_test_s  if needs_scale else X_test

    t0 = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(Xte)
    infer_ms = (time.time()-t0)/len(Xte)*1000

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    try:
        y_prob = model.predict_proba(Xte)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    except:
        auc = float('nan')

    cv_res  = cross_validate(model, Xtr, y_train, cv=cv, scoring='f1_weighted')
    cv_mean = cv_res['test_score'].mean()
    cv_std  = cv_res['test_score'].std()

    feat_imp = None
    if hasattr(model, 'feature_importances_'):
        feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_))

    results.append({
        'model': name, 'accuracy': acc, 'precision_w': prec, 'recall_w': rec,
        'f1_weighted': f1, 'roc_auc': auc, 'cv_f1_mean': cv_mean, 'cv_f1_std': cv_std,
        'train_time_s': train_time, 'infer_time_ms': infer_ms,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'y_pred': y_pred.tolist(), 'y_prob': y_prob.tolist() if hasattr(model,'predict_proba') else None,
        'feat_imp': feat_imp, 'needs_scale': needs_scale,
        'class_report': classification_report(y_test, y_pred,
                                               target_names=le.classes_, output_dict=True)
    })

    dump(model, os.path.join(MODEL_DIR,
         name.replace(' ','_').replace('(','').replace(')','') + '.pkl'))
    print(f"[✓] {name:<25} acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  "
          f"cv={cv_mean:.4f}±{cv_std:.4f}  train={train_time:.2f}s  infer={infer_ms:.4f}ms")

print('\\nAll models trained.')
"""),

md("## Comparison Table"),
code("""\
df_res = pd.DataFrame([{
    'Model': r['model'],
    'Accuracy': round(r['accuracy'],4),
    'Precision': round(r['precision_w'],4),
    'Recall': round(r['recall_w'],4),
    'F1 (weighted)': round(r['f1_weighted'],4),
    'ROC-AUC': round(r['roc_auc'],4) if not (isinstance(r['roc_auc'],float) and np.isnan(r['roc_auc'])) else 'N/A',
    'CV F1 Mean': round(r['cv_f1_mean'],4),
    'CV F1 Std': round(r['cv_f1_std'],4),
    'Train (s)': round(r['train_time_s'],3),
    'Infer (ms/pt)': round(r['infer_time_ms'],4),
} for r in results])

df_res = df_res.sort_values('F1 (weighted)', ascending=False).reset_index(drop=True)
df_res.to_csv('../ml/model_comparison.csv', index=False)
print(df_res.to_string(index=False))
"""),

md("## Bar Chart — F1 & Accuracy Comparison"),
code("""\
df_plot = df_res.set_index('Model')[['Accuracy','F1 (weighted)','CV F1 Mean']].astype(float)

ax = df_plot.plot(kind='bar', figsize=(13, 5), edgecolor='white',
                  color=['#4a9eff','#ff6b6b','#69db7c'], width=0.7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison — Accuracy / F1 / CV-F1')
ax.axhline(0.9, color='white', linestyle='--', alpha=0.3, linewidth=1)
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=7, padding=2)
plt.tight_layout()
plt.savefig('../ml/comparison_bar.png', dpi=150)
plt.show()
"""),

md("## Confusion Matrices — All Models"),
code("""\
n = len(results)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, r in enumerate(results):
    cm = np.array(r['confusion_matrix'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(r['model'])

plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.savefig('../ml/confusion_matrices.png', dpi=150)
plt.show()
"""),

md("## Feature Importance (Random Forest & XGBoost)"),
code("""\
tree_models = [r for r in results if r['feat_imp'] is not None]

fig, axes = plt.subplots(1, len(tree_models), figsize=(16, 7))
if len(tree_models)==1: axes=[axes]

for ax, r in zip(axes, tree_models):
    fi   = pd.Series(r['feat_imp']).sort_values(ascending=True).tail(15)
    colors_bar = ['#4a9eff'] * len(fi)
    ax.barh(fi.index, fi.values, color=colors_bar, edgecolor='white')
    ax.set_title(f"Feature Importance\\n{r['model']}")
    ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig('../ml/feature_importance.png', dpi=150)
plt.show()
"""),

md("## Per-Class F1 Heatmap"),
code("""\
class_f1 = {}
for r in results:
    row = {}
    for cls in le.classes_:
        row[cls] = round(r['class_report'].get(cls, {}).get('f1-score', 0), 3)
    class_f1[r['model']] = row

df_cf1 = pd.DataFrame(class_f1).T
plt.figure(figsize=(8,5))
sns.heatmap(df_cf1.astype(float), annot=True, fmt='.3f', cmap='YlGn',
            linewidths=0.5, vmin=0, vmax=1)
plt.title('Per-Class F1 Score per Model')
plt.tight_layout()
plt.savefig('../ml/per_class_f1.png', dpi=150)
plt.show()
print(df_cf1.to_string())
"""),

md("## Inference Time vs Accuracy (Trade-off)"),
code("""\
fig, ax = plt.subplots(figsize=(9,6))
for r in results:
    ax.scatter(r['infer_time_ms'], r['accuracy'], s=120, zorder=5)
    ax.annotate(r['model'], (r['infer_time_ms'], r['accuracy']),
                textcoords='offset points', xytext=(6,4), fontsize=8)
ax.set_xlabel('Inference Time (ms / point)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Inference Time Trade-off')
plt.tight_layout()
plt.savefig('../ml/tradeoff_plot.png', dpi=150)
plt.show()
"""),

md("## Select & Save Best Model"),
code("""\
best    = max(results, key=lambda r: r['f1_weighted'])
print(f"★  Best model: {best['model']}")
print(f"   Accuracy  : {best['accuracy']:.4f}")
print(f"   F1-W      : {best['f1_weighted']:.4f}")
print(f"   ROC-AUC   : {best['roc_auc']:.4f}")
print(f"   CV F1     : {best['cv_f1_mean']:.4f} ± {best['cv_f1_std']:.4f}")
print(f"   Infer     : {best['infer_time_ms']:.4f} ms/pt")

with open(os.path.join(MODEL_DIR,'best_model.txt'),'w') as f:
    f.write(best['model'])

# Save full results JSON
full = [{k:v for k,v in r.items() if k not in ('y_pred','y_prob')} for r in results]
with open('../ml/model_results.json','w') as f:
    json.dump(full, f, indent=2)

print(f"\\nBest model name saved → models/best_model.txt")
print(f"Full results        saved → ml/model_results.json")
print(f"Comparison CSV      saved → ml/model_comparison.csv")
"""),

md("## Full Classification Report — Best Model"),
code("""\
print(f"=== {best['model']} ===\\n")
print(classification_report(
    y_test,
    np.array(best['y_pred']),
    target_names=le.classes_
))
"""),
]

# Write notebooks
nb1_path = os.path.join(ROOT, "ml", "01_explore.ipynb")
nb2_path = os.path.join(ROOT, "ml", "02_train_compare.ipynb")

with open(nb1_path, "w") as f:
    json.dump(nb(nb1_cells), f, indent=1)

with open(nb2_path, "w") as f:
    json.dump(nb(nb2_cells), f, indent=1)

print(f"[✓] Created: {nb1_path}")
print(f"[✓] Created: {nb2_path}")
