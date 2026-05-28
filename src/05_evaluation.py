"""
05_evaluation.py — Évaluation complète (6 modèles)
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings("ignore")

print("="*60)
print("ÉTAPE 5 — ÉVALUATION COMPLÈTE")
print("="*60)

y_test = np.load("data/y_test.npy")
with open("data/results.json") as f:
    results = json.load(f)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "Naïve Bayes":     "#378ADD",
    "SVM":             "#1D9E75",
    "Random Forest":   "#EF9F27",
    "LSTM":            "#D4537E",
    "BERT":            "#534AB7",
    "Twitter-RoBERTa": "#2E8B57"
}

# ── 1. Tableau métriques ──────────────────────────────────────
print("\n[1/4] Tableau des métriques...")
rows = []
for name, res in results.items():
    r = res["report"]
    rows.append({
        "Modèle":      name,
        "Accuracy":    round(r["accuracy"],4),
        "Précision":   round(r["weighted avg"]["precision"],4),
        "Rappel":      round(r["weighted avg"]["recall"],4),
        "F1-score":    round(r["weighted avg"]["f1-score"],4),
        "F1 Négatif":  round(r.get("0",{}).get("f1-score",0),4),
        "F1 Positif":  round(r.get("1",{}).get("f1-score",0),4),
    })
df_m = pd.DataFrame(rows).set_index("Modèle")
print("\n" + df_m.to_string())
df_m.to_csv("results/metrics_comparison.csv")

# ── 2. Matrices de confusion ──────────────────────────────────
print("\n[2/4] Matrices de confusion...")
n_models = len(results)
ncols = 3
nrows = (n_models + ncols - 1) // ncols

fig = plt.figure(figsize=(20, 6*nrows))
fig.suptitle("Matrices de Confusion — Comparaison de tous les modèles",
             fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                       hspace=0.45, wspace=0.35)

for idx, (name, res) in enumerate(results.items()):
    ax  = fig.add_subplot(gs[idx//ncols, idx%ncols])
    yp  = np.array(res["y_pred"])
    yt  = np.array(res["y_true"])[:len(yp)]
    cm  = confusion_matrix(yt, yp)
    pct = cm.astype(float)/cm.sum(axis=1, keepdims=True)*100

    sns.heatmap(pct, annot=False, ax=ax, cmap="Blues",
                linewidths=0.5, linecolor="white",
                xticklabels=["Négatif","Positif"],
                yticklabels=["Négatif","Positif"],
                cbar=False, vmin=0, vmax=100)

    for i in range(2):
        for j in range(2):
            color = "white" if pct[i,j] > 55 else "black"
            ax.text(j+0.5, i+0.38, f"{pct[i,j]:.1f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
            ax.text(j+0.5, i+0.65, f"n={cm[i,j]:,}",
                    ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_title(f"{name}\nF1 = {res['f1']:.4f}",
                 fontweight="bold", color=COLORS.get(name,"#333"))
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")

# Masquer subplots vides
for i in range(n_models, nrows*ncols):
    fig.add_subplot(gs[i//ncols, i%ncols]).set_visible(False)

plt.savefig("results/figures/05_confusion_matrices.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/05_confusion_matrices.png")

# ── 3. Comparaison métriques ──────────────────────────────────
print("\n[3/4] Comparaison des métriques...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Comparaison des Modèles — Métriques",
             fontsize=14, fontweight="bold")

model_names = list(df_m.index)
bar_colors  = [COLORS.get(n,"#888") for n in model_names]
x = np.arange(len(model_names))
w = 0.35

for ax_i, (m1, m2) in enumerate([
    ("Accuracy","F1-score"),
    ("F1 Négatif","F1 Positif")
]):
    ax = axes[ax_i]
    b1 = ax.bar(x-w/2, df_m[m1], w, color=bar_colors,
                alpha=0.85, edgecolor="white", label=m1)
    b2 = ax.bar(x+w/2, df_m[m2], w, color=bar_colors,
                alpha=0.5, edgecolor="white", label=m2)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0.6, 0.87)
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_title(f"{m1} vs {m2}", fontweight="bold")
    for b in [b1, b2]:
        for bar in b:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.003,
                    f"{bar.get_height():.3f}",
                    ha="center", fontsize=7, fontweight="bold")

plt.tight_layout()
plt.savefig("results/figures/05_metrics_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/05_metrics_comparison.png")

# ── 4. Heatmap ────────────────────────────────────────────────
print("\n[4/4] Heatmap des métriques...")
fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(
    df_m[["Accuracy","Précision","Rappel","F1-score","F1 Négatif","F1 Positif"]],
    annot=True, fmt=".3f", cmap="YlGn",
    linewidths=0.5, ax=ax, vmin=0.70, vmax=0.85,
    annot_kws={"size":12,"weight":"bold"}
)
ax.set_title("Heatmap des métriques par modèle",
             fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("results/figures/05_metrics_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/05_metrics_heatmap.png")

# ── Analyse d'erreurs ─────────────────────────────────────────
print("\n[BONUS] Analyse d'erreurs Twitter-RoBERTa...")
test_df = pd.read_csv("data/test.csv").fillna("")
yp_rob  = np.array(results["Twitter-RoBERTa"]["y_pred"])
yt_rob  = np.array(results["Twitter-RoBERTa"]["y_true"])[:len(yp_rob)]
errors  = np.where(yp_rob != yt_rob)[0]
print(f"  Erreurs : {len(errors)} / {len(yt_rob)} "
      f"({len(errors)/len(yt_rob)*100:.1f}%)")

print("\n✓ Étape 5 terminée.\n")
