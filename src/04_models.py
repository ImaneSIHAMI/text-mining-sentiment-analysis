"""
04_models.py — NB · SVM · RF · LSTM (PyTorch) · BERT (PyTorch)
"""
import os, pickle, json, warnings
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings("ignore")

print("="*60)
print("ÉTAPE 4 — MODÉLISATION")
print("="*60)

y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")
Xtr_tfidf = sparse.load_npz("data/X_train_tfidf.npz")
Xte_tfidf = sparse.load_npz("data/X_test_tfidf.npz")
Xtr_ng    = sparse.load_npz("data/X_train_ng.npz")
Xte_ng    = sparse.load_npz("data/X_test_ng.npz")
print(f"\nDonnées : {len(y_train):,} train | {len(y_test):,} test")

results = {}

# ── 1. Naïve Bayes ────────────────────────────────────────────
print("\n[1/5] Naïve Bayes Multinomial...")
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=0.1)
nb.fit(Xtr_tfidf, y_train)
yp = nb.predict(Xte_tfidf)
f1 = f1_score(y_test, yp, average="weighted")
results["Naïve Bayes"] = {
    "f1": round(f1,4),
    "report": classification_report(y_test, yp, output_dict=True),
    "y_pred": yp.tolist(), "y_true": y_test.tolist()
}
with open("data/model_nb.pkl","wb") as f_: pickle.dump(nb, f_)
print(f"  F1 (weighted) = {f1:.4f}")

# ── 2. SVM ────────────────────────────────────────────────────
print("\n[2/5] SVM LinearSVC...")
from sklearn.svm import LinearSVC
svm = LinearSVC(C=1.0, max_iter=3000, random_state=42)
svm.fit(Xtr_tfidf, y_train)
yp = svm.predict(Xte_tfidf)
f1 = f1_score(y_test, yp, average="weighted")
results["SVM"] = {
    "f1": round(f1,4),
    "report": classification_report(y_test, yp, output_dict=True),
    "y_pred": yp.tolist(), "y_true": y_test.tolist()
}
with open("data/model_svm.pkl","wb") as f_: pickle.dump(svm, f_)
print(f"  F1 (weighted) = {f1:.4f}")

# ── 3. Random Forest ──────────────────────────────────────────
print("\n[3/5] Random Forest (300 arbres, N-grams)...")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                             random_state=42, class_weight="balanced")
rf.fit(Xtr_ng, y_train)
yp = rf.predict(Xte_ng)
f1 = f1_score(y_test, yp, average="weighted")
results["Random Forest"] = {
    "f1": round(f1,4),
    "report": classification_report(y_test, yp, output_dict=True),
    "y_pred": yp.tolist(), "y_true": y_test.tolist()
}
with open("data/model_rf.pkl","wb") as f_: pickle.dump(rf, f_)
print(f"  F1 (weighted) = {f1:.4f}")

# ── 4. LSTM (PyTorch) ─────────────────────────────────────────
print("\n[4/5] LSTM Bidirectionnel (PyTorch)...")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec as GW2V

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {device}")

MAX_LEN = 80
EMBED   = 200
BATCH   = 256
EPOCHS  = 12
LR      = 3e-4

train_df = pd.read_csv("data/train.csv").fillna("")
test_df  = pd.read_csv("data/test.csv").fillna("")

w2v      = GW2V.load("data/word2vec.model")
word2idx = {w: i+1 for i, w in enumerate(w2v.wv.index_to_key)}
VOCAB    = len(word2idx) + 1

def to_seq(texts, max_len):
    seqs = []
    for t in texts:
        ids = [word2idx.get(w, 0) for w in str(t).split()][:max_len]
        ids += [0] * (max_len - len(ids))
        seqs.append(ids)
    return np.array(seqs, dtype=np.int64)

Xtr_seq = to_seq(train_df["text"], MAX_LEN)
Xte_seq = to_seq(test_df["text"],  MAX_LEN)

# Matrice embeddings — TRAINABLE cette fois
embed_mat = np.zeros((VOCAB, EMBED))
for word, idx in word2idx.items():
    if word in w2v.wv:
        embed_mat[idx] = w2v.wv[word]

class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

tr_loader = DataLoader(TweetDataset(Xtr_seq, y_train),
                       batch_size=BATCH, shuffle=True,  num_workers=0)
te_loader = DataLoader(TweetDataset(Xte_seq, y_test),
                       batch_size=BATCH, shuffle=False, num_workers=0)

class BiLSTM(nn.Module):
    def __init__(self, vocab, embed_dim, embed_matrix):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
        # Initialiser avec Word2Vec MAIS laisser entraînable
        self.emb.weight.data.copy_(
            torch.tensor(embed_matrix, dtype=torch.float32))
        self.lstm1 = nn.LSTM(embed_dim, 128, batch_first=True,
                             bidirectional=True, dropout=0.3,
                             num_layers=1)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True)
        self.drop  = nn.Dropout(0.3)
        self.fc1   = nn.Linear(64, 64)
        self.fc2   = nn.Linear(64, 1)
        self.relu  = nn.ReLU()
        self.bn    = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm1(x)
        x = self.drop(x)
        x, (hn, _) = self.lstm2(x)
        # Utiliser le dernier hidden state
        x = hn[-1]
        x = self.drop(x)
        x = self.bn(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(1)

model     = BiLSTM(VOCAB, EMBED, embed_mat).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, steps_per_epoch=len(tr_loader), epochs=EPOCHS)
criterion = nn.BCEWithLogitsLoss()

history = {"loss":[], "val_loss":[], "accuracy":[], "val_accuracy":[]}
best_val_loss = float("inf")
patience_cnt  = 0
PATIENCE      = 3

print(f"  Entraînement sur {len(train_df):,} tweets...")
for epoch in range(EPOCHS):
    # ── Train
    model.train()
    tr_loss = tr_correct = tr_total = 0
    for Xb, yb in tr_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        tr_loss    += loss.item() * len(yb)
        preds       = (torch.sigmoid(out) > 0.5).float()
        tr_correct += (preds == yb).sum().item()
        tr_total   += len(yb)

    # ── Validation
    model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for Xb, yb in te_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out  = model(Xb)
            loss = criterion(out, yb)
            val_loss    += loss.item() * len(yb)
            preds        = (torch.sigmoid(out) > 0.5).float()
            val_correct += (preds == yb).sum().item()
            val_total   += len(yb)

    tl = tr_loss/tr_total;   ta = tr_correct/tr_total
    vl = val_loss/val_total; va = val_correct/val_total
    history["loss"].append(tl);       history["accuracy"].append(ta)
    history["val_loss"].append(vl);   history["val_accuracy"].append(va)
    print(f"  Epoch {epoch+1}/{EPOCHS} — "
          f"loss={tl:.4f} acc={ta:.4f} "
          f"val_loss={vl:.4f} val_acc={va:.4f}")

    # Early stopping
    if vl < best_val_loss:
        best_val_loss = vl
        torch.save(model.state_dict(), "data/model_lstm_best.pt")
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping à l'époque {epoch+1}")
            break

# Charger le meilleur modèle
model.load_state_dict(torch.load("data/model_lstm_best.pt",
                                  map_location=device))
model.eval()
all_preds = []
with torch.no_grad():
    for Xb, _ in te_loader:
        out   = model(Xb.to(device))
        preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
        all_preds.extend(preds)

yp = np.array(all_preds)
f1 = f1_score(y_test, yp, average="weighted")
results["LSTM"] = {
    "f1": round(f1,4),
    "report": classification_report(y_test, yp, output_dict=True),
    "y_pred": yp.tolist(), "y_true": y_test.tolist(),
    "history": history
}
torch.save(model.state_dict(), "data/model_lstm.pt")
print(f"  F1 (weighted) = {f1:.4f}")

# ── 5. BERT (PyTorch HuggingFace Trainer) ─────────────────────
print("\n[5/5] BERT fine-tuning (DistilBERT, PyTorch)...")
from transformers import (DistilBertTokenizerFast,
                           DistilBertForSequenceClassification,
                           TrainingArguments, Trainer,
                           TrainerCallback)

BERT_NAME = "distilbert-base-uncased"
MAX_B     = 128
N_TRAIN   = 20_000
N_TEST    = 5_000
EPOCHS_B  = 3

tok_bert = DistilBertTokenizerFast.from_pretrained(BERT_NAME)

rng    = np.random.default_rng(42)
idx_tr = rng.choice(len(train_df), min(N_TRAIN, len(train_df)), replace=False)
idx_te = rng.choice(len(test_df),  min(N_TEST,  len(test_df)),  replace=False)

texts_tr = train_df["text"].iloc[idx_tr].tolist()
texts_te = test_df["text"].iloc[idx_te].tolist()
y_tr_b   = y_train[idx_tr].astype(int)
y_te_b   = y_test[idx_te].astype(int)

enc_tr = tok_bert(texts_tr, truncation=True, padding=True,
                  max_length=MAX_B, return_tensors="pt")
enc_te = tok_bert(texts_te, truncation=True, padding=True,
                  max_length=MAX_B, return_tensors="pt")

class BertDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc    = enc
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item

ds_tr = BertDS(enc_tr, y_tr_b)
ds_te = BertDS(enc_te, y_te_b)

bert_model = DistilBertForSequenceClassification.from_pretrained(
    BERT_NAME, num_labels=2)

use_gpu = torch.cuda.is_available()

args = TrainingArguments(
    output_dir          = "data/bert_checkpoints",
    num_train_epochs    = EPOCHS_B,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 64,
    learning_rate       = 2e-5,
    eval_strategy       = "epoch",
    save_strategy       = "no",
    logging_steps       = 50,
    report_to           = "none",
    use_cpu             = not use_gpu,
    dataloader_num_workers = 0,
)

bert_hist = {"loss":[], "val_loss":[], "accuracy":[], "val_accuracy":[]}

class LogCB(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            bert_hist["val_loss"].append(
                round(metrics.get("eval_loss", 0), 4))

trainer = Trainer(
    model         = bert_model,
    args          = args,
    train_dataset = ds_tr,
    eval_dataset  = ds_te,
    callbacks     = [LogCB()]
)
trainer.train()

preds_out = trainer.predict(ds_te)
yp_b = np.argmax(preds_out.predictions, axis=1)
f1_b = f1_score(y_te_b, yp_b, average="weighted")

# Reconstruire history depuis les logs
train_logs = [x for x in trainer.state.log_history
              if "loss" in x and "eval_loss" not in x]
eval_logs  = [x for x in trainer.state.log_history if "eval_loss" in x]
bert_hist["loss"]         = [round(x["loss"],4) for x in train_logs]
bert_hist["val_loss"]     = [round(x["eval_loss"],4) for x in eval_logs]
bert_hist["accuracy"]     = [0]*len(bert_hist["loss"])
bert_hist["val_accuracy"] = [round(x.get("eval_accuracy",0),4)
                              for x in eval_logs]

results["BERT"] = {
    "f1": round(f1_b, 4),
    "report": classification_report(y_te_b, yp_b, output_dict=True),
    "y_pred": yp_b.tolist(), "y_true": y_te_b.tolist(),
    "history": bert_hist
}
bert_model.save_pretrained("data/model_bert")
tok_bert.save_pretrained("data/model_bert")
print(f"  F1 (weighted) = {f1_b:.4f}")

# ── Résumé ────────────────────────────────────────────────────
print("\n" + "="*40)
print("RÉSUMÉ DES PERFORMANCES")
print("="*40)
for name, res in results.items():
    print(f"  {name:<15} F1 = {res['f1']:.4f}")

with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n  -> data/results.json")
print("\n OK Etape 4 terminee.\n")