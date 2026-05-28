"""
04b_twitter_roberta.py
Ajout de Twitter-RoBERTa comme 6ème modèle
Lance APRES 04_models.py — met à jour results.json
"""
import json, warnings
import numpy as np, pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report
from transformers import (AutoTokenizer,
                           AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, TrainerCallback)
warnings.filterwarnings("ignore")

print("="*60)
print("MODÈLE 6 — Twitter-RoBERTa (cardiffnlp)")
print("="*60)

# ── Chargement résultats existants ────────────────────────────
with open("data/results.json") as f:
    results = json.load(f)

y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")
train_df = pd.read_csv("data/train.csv").fillna("")
test_df  = pd.read_csv("data/test.csv").fillna("")

# ── Paramètres ────────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN    = 128
N_TRAIN    = 30_000   # plus de données que DistilBERT
N_TEST     = 5_000
EPOCHS     = 3
BATCH      = 32
LR         = 2e-5

print(f"\nModèle    : {MODEL_NAME}")
print(f"Train     : {N_TRAIN:,} tweets")
print(f"Test      : {N_TEST:,} tweets")
print(f"Époques   : {EPOCHS}")
print(f"GPU       : {torch.cuda.is_available()}")

# ── Tokenisation ──────────────────────────────────────────────
print("\n[1/3] Téléchargement et tokenisation...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

rng    = np.random.default_rng(123)
idx_tr = rng.choice(len(train_df), min(N_TRAIN, len(train_df)), replace=False)
idx_te = rng.choice(len(test_df),  min(N_TEST,  len(test_df)),  replace=False)

texts_tr = train_df["text"].iloc[idx_tr].tolist()
texts_te = test_df["text"].iloc[idx_te].tolist()
y_tr     = y_train[idx_tr].astype(int)
y_te     = y_test[idx_te].astype(int)

enc_tr = tokenizer(texts_tr, truncation=True, padding=True,
                   max_length=MAX_LEN, return_tensors="pt")
enc_te = tokenizer(texts_te, truncation=True, padding=True,
                   max_length=MAX_LEN, return_tensors="pt")

class TweetDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc    = enc
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item

ds_tr = TweetDS(enc_tr, y_tr)
ds_te = TweetDS(enc_te, y_te)

# ── Modèle ────────────────────────────────────────────────────
print("\n[2/3] Fine-tuning Twitter-RoBERTa...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)

use_gpu = torch.cuda.is_available()

args = TrainingArguments(
    output_dir          = "data/roberta_checkpoints",
    num_train_epochs    = EPOCHS,
    per_device_train_batch_size = BATCH,
    per_device_eval_batch_size  = 64,
    learning_rate       = LR,
    eval_strategy       = "epoch",
    save_strategy       = "no",
    logging_steps       = 50,
    report_to           = "none",
    use_cpu             = not use_gpu,
    dataloader_num_workers = 0,
    warmup_ratio        = 0.1,
    weight_decay        = 0.01,
)

roberta_hist = {"loss":[], "val_loss":[], "accuracy":[], "val_accuracy":[]}

class LogCB(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            roberta_hist["val_loss"].append(
                round(metrics.get("eval_loss", 0), 4))

trainer = Trainer(
    model         = model,
    args          = args,
    train_dataset = ds_tr,
    eval_dataset  = ds_te,
    callbacks     = [LogCB()]
)
trainer.train()

# ── Évaluation ────────────────────────────────────────────────
print("\n[3/3] Évaluation...")
preds_out = trainer.predict(ds_te)
yp = np.argmax(preds_out.predictions, axis=1)
f1 = f1_score(y_te, yp, average="weighted")

# Reconstruire history
train_logs = [x for x in trainer.state.log_history
              if "loss" in x and "eval_loss" not in x]
eval_logs  = [x for x in trainer.state.log_history if "eval_loss" in x]
roberta_hist["loss"]         = [round(x["loss"],4) for x in train_logs]
roberta_hist["val_loss"]     = [round(x["eval_loss"],4) for x in eval_logs]
roberta_hist["val_accuracy"] = [round(x.get("eval_accuracy",0),4)
                                 for x in eval_logs]
roberta_hist["accuracy"]     = [0]*len(roberta_hist["loss"])

# ── Mise à jour results.json ──────────────────────────────────
results["Twitter-RoBERTa"] = {
    "f1": round(f1, 4),
    "report": classification_report(y_te, yp, output_dict=True),
    "y_pred": yp.tolist(),
    "y_true": y_te.tolist(),
    "history": roberta_hist
}

model.save_pretrained("data/model_roberta")
tokenizer.save_pretrained("data/model_roberta")

with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)

# ── Résumé ────────────────────────────────────────────────────
print("\n" + "="*50)
print("RÉSUMÉ COMPLET — TOUS LES MODÈLES")
print("="*50)
for name, res in results.items():
    marker = " ← MEILLEUR" if res["f1"] == max(r["f1"] for r in results.values()) else ""
    print(f"  {name:<20} F1 = {res['f1']:.4f}{marker}")

print(f"\n  Twitter-RoBERTa F1 = {f1:.4f}")
print("\n  -> data/results.json mis à jour")
print("\n OK Twitter-RoBERTa terminé !\n")
