"""
03_representation.py — BoW · TF-IDF · N-grams · Word2Vec
"""
import os, pickle, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse
from gensim.models import Word2Vec
warnings.filterwarnings("ignore")

print("="*60)
print("ÉTAPE 3 — REPRÉSENTATION VECTORIELLE")
print("="*60)

df = pd.read_csv("data/preprocessed.csv").dropna(subset=["final_text"])
df["final_text"] = df["final_text"].fillna("").astype(str)
X, y = df["final_text"], df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pd.DataFrame({"text":X_train,"sentiment":y_train}).to_csv("data/train.csv",index=False)
pd.DataFrame({"text":X_test, "sentiment":y_test }).to_csv("data/test.csv", index=False)
np.save("data/y_train.npy", y_train.values)
np.save("data/y_test.npy",  y_test.values)
print(f"\nTrain : {len(X_train):,} | Test : {len(X_test):,}")

# ── BoW ───────────────────────────────────────────────────────
print("\n[1/4] Bag of Words...")
bow = CountVectorizer(max_features=20_000, min_df=3, max_df=0.95)
Xtr_bow = bow.fit_transform(X_train)
Xte_bow = bow.transform(X_test)
sparse.save_npz("data/X_train_bow.npz", Xtr_bow)
sparse.save_npz("data/X_test_bow.npz",  Xte_bow)
with open("data/vec_bow.pkl","wb") as f: pickle.dump(bow,f)
print(f"  Vocabulaire : {len(bow.vocabulary_):,} | Dim : {Xtr_bow.shape[1]:,}")

# ── TF-IDF ────────────────────────────────────────────────────
print("\n[2/4] TF-IDF unigrammes...")
tfidf = TfidfVectorizer(max_features=20_000, min_df=3, max_df=0.95, sublinear_tf=True)
Xtr_tfidf = tfidf.fit_transform(X_train)
Xte_tfidf = tfidf.transform(X_test)
sparse.save_npz("data/X_train_tfidf.npz", Xtr_tfidf)
sparse.save_npz("data/X_test_tfidf.npz",  Xte_tfidf)
with open("data/vec_tfidf.pkl","wb") as f: pickle.dump(tfidf,f)
print(f"  Vocabulaire : {len(tfidf.vocabulary_):,} | Dim : {Xtr_tfidf.shape[1]:,}")

# ── N-grams ───────────────────────────────────────────────────
print("\n[3/4] TF-IDF N-grams (1,2,3)...")
ngrams = TfidfVectorizer(max_features=30_000, ngram_range=(1,3),
                         min_df=3, max_df=0.95, sublinear_tf=True)
Xtr_ng = ngrams.fit_transform(X_train)
Xte_ng = ngrams.transform(X_test)
sparse.save_npz("data/X_train_ng.npz", Xtr_ng)
sparse.save_npz("data/X_test_ng.npz",  Xte_ng)
with open("data/vec_ng.pkl","wb") as f: pickle.dump(ngrams,f)
print(f"  Vocabulaire : {len(ngrams.vocabulary_):,} | Dim : {Xtr_ng.shape[1]:,}")

# ── Word2Vec ──────────────────────────────────────────────────
print("\n[4/4] Word2Vec (Skip-Gram, dim=200)...")
sentences = [t.split() for t in X_train]
w2v = Word2Vec(sentences, vector_size=200, window=5,
               min_count=3, workers=4, epochs=10, sg=1, seed=42)
w2v.save("data/word2vec.model")
print(f"  Vocabulaire : {len(w2v.wv):,} mots | Dim : 200")

def avg_w2v(text, model):
    vecs = [model.wv[w] for w in str(text).split() if w in model.wv]
    return np.mean(vecs,axis=0) if vecs else np.zeros(model.vector_size)

print("  Calcul des vecteurs moyens...")
Xtr_w2v = np.vstack([avg_w2v(t,w2v) for t in X_train])
Xte_w2v = np.vstack([avg_w2v(t,w2v) for t in X_test])
np.save("data/X_train_w2v.npy", Xtr_w2v)
np.save("data/X_test_w2v.npy",  Xte_w2v)

# ── Visualisation ─────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(17,6))
fig.suptitle("Comparaison des méthodes de représentation vectorielle",
             fontsize=14, fontweight="bold")

methods = ["BoW","TF-IDF","N-grams\n(1-3)","Word2Vec"]
vocab   = [len(bow.vocabulary_), len(tfidf.vocabulary_),
           len(ngrams.vocabulary_), len(w2v.wv)]
dims    = [Xtr_bow.shape[1], Xtr_tfidf.shape[1], Xtr_ng.shape[1], 200]
colors  = ["#378ADD","#1D9E75","#EF9F27","#D4537E"]

axes[0].bar(methods, vocab, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Taille du vocabulaire", fontweight="bold")
axes[0].set_ylabel("Nb de termes")
for i,v in enumerate(vocab):
    axes[0].text(i, v+300, f"{v:,}", ha="center", fontsize=9, fontweight="bold")

axes[1].bar(methods, dims, color=colors, edgecolor="white", linewidth=1.5)
axes[1].set_title("Dimension du vecteur résultant", fontweight="bold")
axes[1].set_ylabel("Dimension")
for i,v in enumerate(dims):
    axes[1].text(i, v+300, f"{v:,}", ha="center", fontsize=9, fontweight="bold")

# Densité BoW vs TF-IDF
density_bow   = Xtr_bow.nnz   / (Xtr_bow.shape[0]   * Xtr_bow.shape[1])   * 100
density_tfidf = Xtr_tfidf.nnz / (Xtr_tfidf.shape[0] * Xtr_tfidf.shape[1]) * 100
axes[2].bar(["BoW","TF-IDF","Word2Vec\n(dense)"],
            [density_bow, density_tfidf, 100],
            color=["#378ADD","#1D9E75","#D4537E"],
            edgecolor="white", linewidth=1.5)
axes[2].set_title("Densité de la matrice (%)", fontweight="bold")
axes[2].set_ylabel("Densité (%)")
for i,v in enumerate([density_bow, density_tfidf, 100]):
    axes[2].text(i, v+0.5, f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("results/figures/03_representation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → results/figures/03_representation.png")
print("\n✓ Étape 3 terminée.\n")
