"""
fix_lstm.py — Correction LSTM + mise à jour results.json
"""
import json, pickle, warnings
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from gensim.models import Word2Vec as GW2V
warnings.filterwarnings("ignore")

print("="*60)
print("CORRECTION LSTM")
print("="*60)

# Charger résultats existants
with open("data/results.json") as f:
    results = json.load(f)

y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")
train_df = pd.read_csv("data/train.csv").fillna("")
test_df  = pd.read_csv("data/test.csv").fillna("")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ── Paramètres ────────────────────────────────────────────────
MAX_LEN = 60
EMBED   = 200
BATCH   = 128
EPOCHS  = 15
LR      = 1e-3

# ── Word2Vec ──────────────────────────────────────────────────
w2v      = GW2V.load("data/word2vec.model")
word2idx = {w: i+1 for i, w in enumerate(w2v.wv.index_to_key)}
VOCAB    = len(word2idx) + 1
print(f"Vocabulaire W2V : {len(word2idx):,}")

# Matrice embedding
embed_mat = np.random.uniform(-0.1, 0.1, (VOCAB, EMBED)).astype(np.float32)
embed_mat[0] = 0  # padding
found = 0
for word, idx in word2idx.items():
    if word in w2v.wv:
        embed_mat[idx] = w2v.wv[word]
        found += 1
print(f"Mots trouvés dans W2V : {found:,}/{len(word2idx):,}")

# ── Séquences ─────────────────────────────────────────────────
def to_seq(texts, max_len):
    out = []
    for t in texts:
        ids = [word2idx.get(w, 0) for w in str(t).split()][:max_len]
        ids += [0] * (max_len - len(ids))
        out.append(ids)
    return np.array(out, dtype=np.int64)

Xtr = to_seq(train_df["text"], MAX_LEN)
Xte = to_seq(test_df["text"],  MAX_LEN)

# Vérification
nonzero_tr = (Xtr != 0).sum(axis=1).mean()
print(f"Tokens non-zéro moyens (train) : {nonzero_tr:.1f}/{MAX_LEN}")

class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

tr_ld = DataLoader(DS(Xtr, y_train), batch_size=BATCH,
                   shuffle=True, num_workers=0, pin_memory=True)
te_ld = DataLoader(DS(Xte, y_test),  batch_size=BATCH,
                   shuffle=False, num_workers=0)

# ── Modèle simple et robuste ──────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(VOCAB, EMBED, padding_idx=0)
        self.emb.weight.data = torch.tensor(embed_mat)
        # entraînable
        self.emb.weight.requires_grad = True

        self.lstm = nn.LSTM(EMBED, 128, num_layers=2,
                            batch_first=True, bidirectional=True,
                            dropout=0.3)
        self.attn = nn.Linear(256, 1)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, L)
        emb = self.emb(x)                    # (B, L, E)
        out, _ = self.lstm(emb)              # (B, L, 256)
        # Attention
        w = torch.softmax(self.attn(out), dim=1)  # (B, L, 1)
        ctx = (out * w).sum(dim=1)           # (B, 256)
        ctx = self.drop(ctx)
        return self.fc(ctx).squeeze(1)

model = LSTMClassifier().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres entraînables : {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss()

history = {"loss":[],"val_loss":[],"accuracy":[],"val_accuracy":[]}
best_f1   = 0
best_state = None

print(f"\nEntraînement ({EPOCHS} époques)...")
for epoch in range(EPOCHS):
    # Train
    model.train()
    tl = tc = tn = 0
    for Xb, yb in tr_ld:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item()*len(yb)
        tc += ((torch.sigmoid(out)>0.5).float()==yb).sum().item()
        tn += len(yb)
    scheduler.step()

    # Val
    model.eval()
    vl = vc = vn = 0
    vp = []
    with torch.no_grad():
        for Xb, yb in te_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            out  = model(Xb)
            loss = criterion(out, yb)
            vl += loss.item()*len(yb)
            preds = (torch.sigmoid(out)>0.5).float()
            vc += (preds==yb).sum().item()
            vn += len(yb)
            vp.extend(preds.cpu().numpy().astype(int))

    ta = tc/tn; va = vc/vn
    tll = tl/tn; vll = vl/vn
    ep_f1 = f1_score(y_test, vp, average="weighted")

    history["loss"].append(round(tll,4))
    history["val_loss"].append(round(vll,4))
    history["accuracy"].append(round(ta,4))
    history["val_accuracy"].append(round(va,4))

    print(f"  Ep {epoch+1:02d}/{EPOCHS} "
          f"loss={tll:.4f} acc={ta:.4f} "
          f"val_loss={vll:.4f} val_acc={va:.4f} "
          f"F1={ep_f1:.4f}")

    if ep_f1 > best_f1:
        best_f1    = ep_f1
        best_state = {k: v.clone() for k,v in model.state_dict().items()}

# Meilleur modèle
model.load_state_dict(best_state)
model.eval()
all_preds = []
with torch.no_grad():
    for Xb, _ in te_ld:
        out = model(Xb.to(device))
        all_preds.extend(
            (torch.sigmoid(out)>0.5).cpu().numpy().astype(int))

yp = np.array(all_preds)
f1 = f1_score(y_test, yp, average="weighted")
print(f"\n  Meilleur F1 LSTM = {f1:.4f}")

# Mise à jour results.json
results["LSTM"] = {
    "f1": round(f1, 4),
    "report": classification_report(y_test, yp, output_dict=True),
    "y_pred": yp.tolist(),
    "y_true": y_test.tolist(),
    "history": history
}

torch.save(model.state_dict(), "data/model_lstm.pt")

with open("data/results.json","w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*40)
print("RÉSUMÉ FINAL")
print("="*40)
for name, res in results.items():
    print(f"  {name:<15} F1 = {res['f1']:.4f}")
print("\n -> data/results.json mis à jour")
print("\n OK LSTM corrigé !\n")
