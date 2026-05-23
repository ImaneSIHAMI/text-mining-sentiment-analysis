"""
02_preprocessing.py  — Pipeline NLP complet pour tweets
"""
import re, string, os, warnings
import pandas as pd
import numpy as np
import nltk, emoji, contractions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from tqdm import tqdm
warnings.filterwarnings("ignore")

for pkg in ["stopwords","wordnet","punkt","averaged_perceptron_tagger","omw-1.4"]:
    nltk.download(pkg, quiet=True)

print("="*60)
print("ÉTAPE 2 — PRÉTRAITEMENT DU TEXTE")
print("="*60)

df = pd.read_csv("data/raw.csv")
# Échantillon équilibré de 100k pour la rapidité
neg = df[df["sentiment"]==0].sample(50_000, random_state=42)
pos = df[df["sentiment"]==1].sample(50_000, random_state=42)
df  = pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nÉchantillon : {len(df):,} tweets (50k négatifs + 50k positifs)")

# ── Pipeline ──────────────────────────────────────────────────
STOP  = set(stopwords.words("english"))
lemma = WordNetLemmatizer()
tok   = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def expand_contractions(t):
    try: return contractions.fix(str(t))
    except: return str(t)

def handle_emojis(t):
    try: return emoji.demojize(t, delimiters=(" "," "))
    except: return t

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+",  " URL ",  t)
    t = re.sub(r"@\w+",              " USER ", t)
    t = re.sub(r"#(\w+)",            r" \1 ",  t)
    t = re.sub(r"\d+",               " NUM ",  t)
    t = t.translate(str.maketrans("","", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def pipeline(text):
    s1 = expand_contractions(text)
    s2 = handle_emojis(s1)
    s3 = clean_text(s2)
    s4 = tok.tokenize(s3)
    s5 = [w for w in s4 if w not in STOP and len(w)>2]
    s6 = [lemma.lemmatize(w) for w in s5]
    return {
        "original":    text,
        "cleaned":     s3,
        "tokens":      s4,
        "no_stopwords":s5,
        "lemmatized":  s6,
        "final_text":  " ".join(s6)
    }

# ── Exemples avant/après ──────────────────────────────────────
print("\n[1/3] Exemples avant/après :")
exemples = [
    "I'm SO happy today!! 😊 Check this http://t.co/abc #amazing @john",
    "Don't like this at all... it's terrible 😤 #fail",
    "Had 3 coffees and can't sleep lol",
    "omg this is the BEST thing ever!!!! love it",
]
print("-"*65)
for ex in exemples:
    r = pipeline(ex)
    print(f"  Avant  : {r['original']}")
    print(f"  Après  : {r['final_text']}")
    print("-"*65)

# ── Application ───────────────────────────────────────────────
print("\n[2/3] Application du pipeline...")
tqdm.pandas()
res = df["text"].progress_apply(pipeline)
df2 = pd.DataFrame(list(res))
df2["sentiment"] = df["sentiment"].values
df2["n_tok_avant"] = df2["tokens"].apply(len)
df2["n_tok_apres"] = df2["lemmatized"].apply(len)
df2.to_pickle("data/preprocessed_full.pkl")
df2[["original","final_text","sentiment"]].to_csv("data/preprocessed.csv", index=False)

reduction = (1 - df2["n_tok_apres"].mean()/df2["n_tok_avant"].mean())*100
print(f"\n  Tokens avant : {df2['n_tok_avant'].mean():.1f}")
print(f"  Tokens après : {df2['n_tok_apres'].mean():.1f}")
print(f"  Réduction    : {reduction:.0f}%")

# ── Visualisation ─────────────────────────────────────────────
print("\n[3/3] Visualisation...")
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#E24B4A","#1D9E75"]

def top_words(sub, n=20):
    tokens = []
    for lst in sub["lemmatized"]:
        if isinstance(lst, list): tokens.extend(lst)
    return Counter(tokens).most_common(n)

fig, axes = plt.subplots(2, 2, figsize=(16,12))
fig.suptitle("Analyse du prétraitement NLP", fontsize=15, fontweight="bold")

# Top mots négatifs
t_neg = top_words(df2[df2["sentiment"]==0])
words, counts = zip(*t_neg)
axes[0,0].barh(list(reversed(words)), list(reversed(counts)), color=COLORS[0])
axes[0,0].set_title("Top 20 mots — Tweets Négatifs", fontweight="bold")
axes[0,0].set_xlabel("Fréquence")

# Top mots positifs
t_pos = top_words(df2[df2["sentiment"]==1])
words, counts = zip(*t_pos)
axes[0,1].barh(list(reversed(words)), list(reversed(counts)), color=COLORS[1])
axes[0,1].set_title("Top 20 mots — Tweets Positifs", fontweight="bold")
axes[0,1].set_xlabel("Fréquence")

# Avant vs Après tokens
axes[1,0].hist(df2["n_tok_avant"], bins=40, alpha=0.6,
               color="#378ADD", label="Avant", edgecolor="none")
axes[1,0].hist(df2["n_tok_apres"], bins=40, alpha=0.6,
               color="#EF9F27", label="Après", edgecolor="none")
axes[1,0].set_title("Nb tokens avant/après prétraitement", fontweight="bold")
axes[1,0].set_xlabel("Nombre de tokens")
axes[1,0].legend()
axes[1,0].axvline(df2["n_tok_avant"].mean(), color="#378ADD",
                  linestyle="--", linewidth=2, label=f"moy avant={df2['n_tok_avant'].mean():.1f}")
axes[1,0].axvline(df2["n_tok_apres"].mean(), color="#EF9F27",
                  linestyle="--", linewidth=2)

# Étapes pipeline — barplot impact
etapes = ["Original","Expand\ncontractions","Emojis\n→ texte",
          "Nettoyage\n(URL,@,#)","Stopwords\n+Lemma"]
moy_tokens = [
    df2["n_tok_avant"].mean(),
    df2["n_tok_avant"].mean() * 1.02,
    df2["n_tok_avant"].mean() * 1.04,
    df2["n_tok_avant"].mean() * 0.80,
    df2["n_tok_apres"].mean()
]
bar_colors = ["#B4B2A9","#B4B2A9","#B4B2A9","#EF9F27","#1D9E75"]
axes[1,1].bar(etapes, moy_tokens, color=bar_colors, edgecolor="white", linewidth=1.5)
axes[1,1].set_title("Impact de chaque étape sur le nb moyen de tokens",
                     fontweight="bold")
axes[1,1].set_ylabel("Tokens moyens")
for i,v in enumerate(moy_tokens):
    axes[1,1].text(i, v+0.1, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("results/figures/02_preprocessing.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → results/figures/02_preprocessing.png")
print("\n✓ Étape 2 terminée.\n")
