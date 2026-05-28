# Text Mining — Analyse de Sentiment sur Twitter
### Projet DATA_INE2_2026 · Pr. EL ASRI Ikram · SIHAMI Imane

Pipeline complet de Text Mining appliqué à l'analyse de sentiment sur le dataset Sentiment140 (1.6M tweets).

---

## Résultats

| Modèle | Représentation | F1-score |
|---|---|---|
| Naïve Bayes | TF-IDF | 0.7397 |
| SVM | TF-IDF | 0.7438 |
| Random Forest | N-grams | 0.7512 |
| LSTM Bidirectionnel | Word2Vec | 0.7650 |
| DistilBERT | BERT embeddings | 0.7676 |
| **Twitter-RoBERTa** | **RoBERTa embeddings** | **0.7945** ✅ |

---

## Structure du projet

```
text-mining-sentiment-analysis/
├── src/
│   ├── 01_data_understanding.py   # Chargement + analyse exploratoire
│   ├── 02_preprocessing.py        # Pipeline NLP complet
│   ├── 03_representation.py       # BoW, TF-IDF, N-grams, Word2Vec
│   ├── 04_models.py               # NB, SVM, RF, LSTM, DistilBERT
│   ├── 04b_twitter_roberta.py     # Twitter-RoBERTa (6ème modèle)
│   ├── 05_evaluation.py           # Métriques + matrices de confusion
│   └── 06_visualizations.py       # WordCloud, t-SNE, courbes
├── data/                          # Données (non incluses dans le repo)
├── results/figures/               # Visualisations générées
├── report/                        # Rapport LaTeX
│   ├── main.tex
│   └── sections/
├── requirements.txt
└── README.md
```

---

## Dataset

**Sentiment140** — 1.6 million de tweets annotés (positif/négatif).

Téléchargement manuel depuis [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) :
1. Téléchargez `training.1600000.processed.noemoticon.csv`
2. Placez-le dans `data/`
3. Lancez le script de conversion :

```cmd
python -c "import pandas as pd; df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, names=['sentiment','id','date','query','user','text']); df['sentiment'] = df['sentiment'].map({0:0, 4:1}); df[['text','sentiment']].to_csv('data/raw.csv', index=False); print(f'OK : {len(df):,} tweets')"
```

---

## Installation

```cmd
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -c "import nltk; [nltk.download(x) for x in ['stopwords','wordnet','punkt','averaged_perceptron_tagger','omw-1.4']]"
```

---

## Exécution

```cmd
python src/01_data_understanding.py   # ~2 min
python src/02_preprocessing.py        # ~5 min
python src/03_representation.py       # ~3 min
python src/04_models.py               # ~60 min (LSTM + BERT)
python src/04b_twitter_roberta.py     # ~90 min (Twitter-RoBERTa)
python src/05_evaluation.py           # ~2 min
python src/06_visualizations.py       # ~5 min
```

Les figures sont générées dans `results/figures/`.

---

## Pipeline

```
Données brutes (1.6M tweets)
    ↓
Prétraitement NLP (6 étapes : contractions, emojis, nettoyage, tokenisation, stopwords, lemmatisation)
    ↓
Représentation vectorielle (BoW · TF-IDF · N-grams · Word2Vec · BERT/RoBERTa embeddings)
    ↓
Modélisation (Naïve Bayes · SVM · Random Forest · LSTM · DistilBERT · Twitter-RoBERTa)
    ↓
Évaluation (Accuracy · F1-score · Matrices de confusion · Analyse d'erreurs)
    ↓
Visualisation (WordCloud · t-SNE · Courbes d'apprentissage · Heatmap)
```

---

## Technologies

- **Python** 3.10+
- **PyTorch** — LSTM Bidirectionnel
- **HuggingFace Transformers** — DistilBERT, Twitter-RoBERTa
- **scikit-learn** — Naïve Bayes, SVM, Random Forest, métriques
- **NLTK / gensim** — Prétraitement, Word2Vec
- **matplotlib / seaborn / wordcloud** — Visualisations

## Note

Les fichiers de modèles lourds (`model.safetensors`, etc.) ne sont pas inclus dans ce repo.
Relancez les scripts d'entraînement pour les régénérer.
