# Text Mining — Analyse de Sentiment Twitter
## DATA_INE2_2026

## 1. Installation (CMD dans le dossier projet)
```
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -c "import nltk; [nltk.download(x) for x in ['stopwords','wordnet','punkt','averaged_perceptron_tagger','omw-1.4']]"
```

## 2. Exécution dans l'ordre
```
python src/01_data_understanding.py
python src/02_preprocessing.py
python src/03_representation.py
python src/04_models.py
python src/05_evaluation.py
python src/06_visualizations.py
```
