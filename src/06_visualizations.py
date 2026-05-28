"""
06_visualizations.py — WordCloud · t-SNE · Courbes LSTM/BERT · Comparaison finale (6 modèles)
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
warnings.filterwarnings("ignore")

print("="*60)
print("ÉTAPE 6 — VISUALISATIONS AVANCÉES")
print("="*60)

df = pd.read_csv("data/preprocessed.csv").dropna(subset=["final_text"])
df["final_text"] = df["final_text"].fillna("").astype(str)
with open("data/results.json") as f:
    results = json.load(f)

plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. WordClouds ─────────────────────────────────────────────
print("\n[1/6] WordClouds positif vs négatif...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Nuages de Mots — Sentiment140 (après prétraitement)",
             fontsize=15, fontweight="bold")

for ax, (lbl, cmap, title) in zip(axes, [
    (0, "Reds",   "Tweets Négatifs"),
    (1, "Greens", "Tweets Positifs")
]):
    sub  = df[df["sentiment"] == lbl]
    text = " ".join(sub["final_text"].sample(
        n=min(20_000, len(sub)), random_state=42))
    wc = WordCloud(width=900, height=550, background_color="white",
                   colormap=cmap, max_words=200, collocations=False,
                   min_font_size=8).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("results/figures/06_wordclouds.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/06_wordclouds.png")

# ── 2. Courbes LSTM ───────────────────────────────────────────
print("\n[2/6] Courbes d'apprentissage LSTM...")
hist   = results["LSTM"]["history"]
epochs = range(1, len(hist["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Courbes d'apprentissage — LSTM Bidirectionnel",
             fontsize=14, fontweight="bold")

axes[0].plot(epochs, hist["loss"],     "o-", color="#378ADD",
             linewidth=2, markersize=5, label="Train")
axes[0].plot(epochs, hist["val_loss"], "s--", color="#E24B4A",
             linewidth=2, markersize=5, label="Validation")
axes[0].set_title("Fonction de perte", fontweight="bold")
axes[0].set_xlabel("Époque"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs, hist["accuracy"],     "o-", color="#1D9E75",
             linewidth=2, markersize=5, label="Train")
axes[1].plot(epochs, hist["val_accuracy"], "s--", color="#EF9F27",
             linewidth=2, markersize=5, label="Validation")
axes[1].set_title("Précision (Accuracy)", fontweight="bold")
axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/06_lstm_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/06_lstm_curves.png")

# ── 3. Courbes BERT ───────────────────────────────────────────
print("\n[3/6] Courbes d'apprentissage BERT...")
hist_b   = results["BERT"]["history"]
val_loss = hist_b.get("val_loss", [])
n_eval   = len(val_loss)
train_loss_all = hist_b.get("loss", [])
if len(train_loss_all) >= n_eval and n_eval > 0:
    chunk = len(train_loss_all) // n_eval
    train_loss = [float(np.mean(train_loss_all[i*chunk:(i+1)*chunk]))
                  for i in range(n_eval)]
else:
    train_loss = train_loss_all
epochs_b = range(1, n_eval + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Courbes d'apprentissage — BERT Fine-tuning (DistilBERT)",
             fontsize=14, fontweight="bold")

if len(train_loss) == n_eval and n_eval > 0:
    axes[0].plot(epochs_b, train_loss, "o-", color="#534AB7",
                 linewidth=2.5, markersize=8, label="Train")
if n_eval > 0:
    axes[0].plot(epochs_b, val_loss, "s--", color="#D4537E",
                 linewidth=2.5, markersize=8, label="Validation")
axes[0].set_title("Fonction de perte", fontweight="bold")
axes[0].set_xlabel("Époque"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)
if n_eval > 0:
    axes[0].set_xticks(list(epochs_b))

val_acc = hist_b.get("val_accuracy", [])
if len(val_acc) == n_eval and n_eval > 0:
    axes[1].plot(epochs_b, val_acc, "s--", color="#D4537E",
                 linewidth=2.5, markersize=8, label="Validation acc")
    axes[1].set_title("Précision Validation", fontweight="bold")
    axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xticks(list(epochs_b))
else:
    axes[1].axis("off")
    axes[1].text(0.5, 0.5,
                 f"BERT DistilBERT\nF1 = {results['BERT']['f1']:.4f}\n3 époques · lr=2e-5",
                 ha="center", va="center", fontsize=13,
                 transform=axes[1].transAxes,
                 bbox=dict(boxstyle="round", facecolor="#F0E6FF",
                           edgecolor="#534AB7", alpha=0.8))

plt.tight_layout()
plt.savefig("results/figures/06_bert_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/06_bert_curves.png")

# ── 4. Courbes Twitter-RoBERTa ────────────────────────────────
print("\n[4/6] Courbes d'apprentissage Twitter-RoBERTa...")
hist_r   = results["Twitter-RoBERTa"]["history"]
val_loss_r = hist_r.get("val_loss", [])
n_eval_r   = len(val_loss_r)
train_loss_r_all = hist_r.get("loss", [])
if len(train_loss_r_all) >= n_eval_r and n_eval_r > 0:
    chunk = len(train_loss_r_all) // n_eval_r
    train_loss_r = [float(np.mean(train_loss_r_all[i*chunk:(i+1)*chunk]))
                    for i in range(n_eval_r)]
else:
    train_loss_r = train_loss_r_all
epochs_r = range(1, n_eval_r + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Courbes d'apprentissage — Twitter-RoBERTa Fine-tuning",
             fontsize=14, fontweight="bold")

if len(train_loss_r) == n_eval_r and n_eval_r > 0:
    axes[0].plot(epochs_r, train_loss_r, "o-", color="#2E8B57",
                 linewidth=2.5, markersize=8, label="Train")
if n_eval_r > 0:
    axes[0].plot(epochs_r, val_loss_r, "s--", color="#FF6B35",
                 linewidth=2.5, markersize=8, label="Validation")
axes[0].set_title("Fonction de perte", fontweight="bold")
axes[0].set_xlabel("Époque"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)
if n_eval_r > 0:
    axes[0].set_xticks(list(epochs_r))

val_acc_r = hist_r.get("val_accuracy", [])
if len(val_acc_r) == n_eval_r and n_eval_r > 0:
    axes[1].plot(epochs_r, val_acc_r, "s--", color="#FF6B35",
                 linewidth=2.5, markersize=8, label="Validation acc")
    axes[1].set_title("Précision Validation", fontweight="bold")
    axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xticks(list(epochs_r))
else:
    axes[1].axis("off")
    axes[1].text(0.5, 0.5,
                 f"Twitter-RoBERTa\nF1 = {results['Twitter-RoBERTa']['f1']:.4f}\n3 époques · lr=2e-5",
                 ha="center", va="center", fontsize=13,
                 transform=axes[1].transAxes,
                 bbox=dict(boxstyle="round", facecolor="#E8F5E9",
                           edgecolor="#2E8B57", alpha=0.8))

plt.tight_layout()
plt.savefig("results/figures/06_roberta_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/06_roberta_curves.png")

# ── 5. t-SNE Word2Vec ─────────────────────────────────────────
print("\n[5/6] t-SNE des embeddings Word2Vec...")
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

w2v       = Word2Vec.load("data/word2vec.model")
top_words = [w for w, _ in w2v.wv.key_to_index.items()][:300]
vectors   = np.array([w2v.wv[w] for w in top_words])

print("  Calcul t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30,
            max_iter=1000, learning_rate="auto", init="pca")
v2d  = tsne.fit_transform(vectors)

POS_W = {"good","great","love","happy","amazing","best","wonderful",
         "excellent","nice","thanks","awesome","beautiful","perfect"}
NEG_W = {"bad","hate","sad","terrible","awful","worst","horrible",
         "ugly","angry","miss","cry","boring","stupid"}

fig, ax = plt.subplots(figsize=(14, 10))
for i, word in enumerate(top_words):
    if word in POS_W:
        c, s, a, z = "#1D9E75", 150, 1.0, 3
    elif word in NEG_W:
        c, s, a, z = "#E24B4A", 150, 1.0, 3
    else:
        c, s, a, z = "#B4B2A9", 25, 0.35, 1
    ax.scatter(v2d[i, 0], v2d[i, 1], c=c, s=s, alpha=a, zorder=z)
    if word in POS_W | NEG_W:
        ax.annotate(word, (v2d[i, 0]+0.5, v2d[i, 1]+0.5),
                    fontsize=9, fontweight="bold")

ax.set_title("t-SNE — Embeddings Word2Vec (300 mots fréquents)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Composante 1"); ax.set_ylabel("Composante 2")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#1D9E75", label="Mots positifs"),
    Patch(color="#E24B4A", label="Mots négatifs"),
    Patch(color="#B4B2A9", label="Autres mots"),
], fontsize=11)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("results/figures/06_tsne_w2v.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/06_tsne_w2v.png")

# ── 6. Comparaison finale 6 modèles ──────────────────────────
print("\n[6/6] Comparaison finale...")
COLORS_M = {
    "Naïve Bayes":     "#378ADD",
    "SVM":             "#1D9E75",
    "Random Forest":   "#EF9F27",
    "LSTM":            "#D4537E",
    "BERT":            "#534AB7",
    "Twitter-RoBERTa": "#2E8B57"
}
repr_label = {
    "Naïve Bayes":     "TF-IDF",
    "SVM":             "TF-IDF",
    "Random Forest":   "N-grams",
    "LSTM":            "Word2Vec",
    "BERT":            "BERT emb.",
    "Twitter-RoBERTa": "RoBERTa emb."
}

names = list(results.keys())
f1s   = [results[n]["f1"] for n in names]
cols  = [COLORS_M.get(n, "#888") for n in names]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(names, f1s, color=cols, edgecolor="white",
              linewidth=1.5, width=0.55)
ax.set_ylim(0.6, 0.87)
ax.set_title("Comparaison finale — F1-score (weighted) — 6 modèles",
             fontsize=14, fontweight="bold")
ax.set_ylabel("F1-score")
ax.axhline(0.77, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax.tick_params(axis="x", rotation=15)

for bar, val, name in zip(bars, f1s, names):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.003,
            f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(bar.get_x()+bar.get_width()/2,
            0.615, repr_label[name],
            ha="center", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("results/figures/06_final_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/06_final_comparison.png")

print("\n" + "="*60)
print("PIPELINE COMPLET TERMINÉ !")
print("="*60)
import os
for f in sorted(os.listdir("results/figures")):
    if f.endswith(".png"):
        print(f"  OK {f}")
