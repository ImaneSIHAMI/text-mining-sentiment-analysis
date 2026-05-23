"""
01_data_understanding.py
Chargement Sentiment140 + analyse exploratoire complète
"""
import os, re, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

print("="*60)
print("ÉTAPE 1 — CHARGEMENT ET COMPRÉHENSION DES DONNÉES")
print("="*60)

# ── Chargement ────────────────────────────────────────────────
print("\n[1/4] Chargement du dataset Sentiment140...")
df = pd.read_csv("data/raw.csv")
df["text"] = df["text"].astype(str)
print(f"    {len(df):,} tweets chargés depuis data/raw.csv")

# ── Statistiques ──────────────────────────────────────────────
print("\n[2/4] Statistiques générales...")
df["length"]     = df["text"].apply(len)
df["word_count"] = df["text"].apply(lambda x: len(x.split()))

dist = df["sentiment"].value_counts().sort_index()
print(f"\n  Total tweets       : {len(df):,}")
print(f"  Manquants          : {df[['text','sentiment']].isnull().sum().to_dict()}")
print(f"  Négatifs (0)       : {dist[0]:,} ({dist[0]/len(df)*100:.1f}%)")
print(f"  Positifs (1)       : {dist[1]:,} ({dist[1]/len(df)*100:.1f}%)")
print(f"  Longueur moy.      : {df['length'].mean():.1f} chars")
print(f"  Mots moyens        : {df['word_count'].mean():.1f}")

# ── Problèmes ─────────────────────────────────────────────────
print("\n[3/4] Détection des problèmes potentiels...")

def count_pattern(series, pattern):
    return series.apply(lambda x: bool(re.search(pattern, str(x)))).sum()

n_dup  = df["text"].duplicated().sum()
n_url  = count_pattern(df["text"], r"http\S+|www\.\S+")
n_men  = count_pattern(df["text"], r"@\w+")
n_hash = count_pattern(df["text"], r"#\w+")
n_emoj = count_pattern(df["text"], u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]")

print(f"  Doublons     : {n_dup:,}")
print(f"  URLs         : {n_url:,}")
print(f"  @mentions    : {n_men:,}")
print(f"  #hashtags    : {n_hash:,}")
print(f"  Emojis       : {n_emoj:,}")

# ── Visualisation ─────────────────────────────────────────────
print("\n[4/4] Génération des visualisations...")
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#E24B4A", "#1D9E75"]

fig = plt.figure(figsize=(16, 11))
fig.suptitle("Analyse Exploratoire — Sentiment140 (1.6M tweets)",
             fontsize=16, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Distribution classes
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(["Négatif", "Positif"], dist.values, color=COLORS,
               edgecolor="white", linewidth=1.5, width=0.5)
for b, v in zip(bars, dist.values):
    ax1.text(b.get_x() + b.get_width()/2, v + 8000,
             f"{v:,}\n({v/len(df)*100:.0f}%)",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_title("Distribution des classes", fontweight="bold")
ax1.set_ylabel("Nombre de tweets")
ax1.set_ylim(0, max(dist.values) * 1.18)

# Distribution longueur
ax2 = fig.add_subplot(gs[0, 1])
for lbl, col, name in zip([0, 1], COLORS, ["Négatif", "Positif"]):
    ax2.hist(df[df["sentiment"] == lbl]["length"],
             bins=50, alpha=0.65, color=col, label=name, edgecolor="none")
ax2.set_title("Longueur des tweets (chars)", fontweight="bold")
ax2.set_xlabel("Longueur")
ax2.legend()

# Distribution mots
ax3 = fig.add_subplot(gs[0, 2])
for lbl, col, name in zip([0, 1], COLORS, ["Négatif", "Positif"]):
    ax3.hist(df[df["sentiment"] == lbl]["word_count"],
             bins=35, alpha=0.65, color=col, label=name, edgecolor="none")
ax3.set_title("Nombre de mots par tweet", fontweight="bold")
ax3.set_xlabel("Nb mots")
ax3.legend()

# Bruit détecté
ax4 = fig.add_subplot(gs[1, 0])
bruit = {"URLs": n_url, "Mentions (@)": n_men,
         "Hashtags (#)": n_hash, "Emojis": n_emoj, "Doublons": n_dup}
ax4.barh(list(bruit.keys()), list(bruit.values()), color="#378ADD")
ax4.set_title("Éléments de bruit détectés", fontweight="bold")
ax4.set_xlabel("Nombre de tweets")
for i, (k, v) in enumerate(bruit.items()):
    ax4.text(v + 2000, i, f"{v:,}", va="center", fontsize=9)

# Boxplot longueur par classe
ax5 = fig.add_subplot(gs[1, 1])
data_box = [df[df["sentiment"] == 0]["length"].values,
            df[df["sentiment"] == 1]["length"].values]
bp = ax5.boxplot(data_box, patch_artist=True,
                 medianprops=dict(color="white", linewidth=2))
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
ax5.set_xticklabels(["Négatif", "Positif"])
ax5.set_title("Distribution longueur par classe", fontweight="bold")
ax5.set_ylabel("Longueur (chars)")

# Statistiques résumé
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
stats_text = (
    f"RÉSUMÉ DU CORPUS\n\n"
    f"Total tweets    : {len(df):,}\n"
    f"Classe 0 (neg)  : {dist[0]:,}\n"
    f"Classe 1 (pos)  : {dist[1]:,}\n"
    f"Equilibre       : {dist[0]/dist[1]:.2f}\n\n"
    f"Longueur moy.   : {df['length'].mean():.1f} chars\n"
    f"Longueur min    : {df['length'].min()} chars\n"
    f"Longueur max    : {df['length'].max()} chars\n\n"
    f"Mots moyens     : {df['word_count'].mean():.1f}\n\n"
    f"Doublons        : {n_dup:,}\n"
    f"Tweets avec URL : {n_url:,} ({n_url/len(df)*100:.1f}%)\n"
    f"Tweets avec @   : {n_men:,} ({n_men/len(df)*100:.1f}%)\n"
    f"Tweets avec #   : {n_hash:,} ({n_hash/len(df)*100:.1f}%)"
)
ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4FF",
                   edgecolor="#378ADD", alpha=0.8))

plt.savefig("results/figures/01_data_understanding.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  -> results/figures/01_data_understanding.png")
print("\n OK Etape 1 terminee.\n")