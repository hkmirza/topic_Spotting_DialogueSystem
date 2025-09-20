
# BiLSTM + Self-Attention for topic spotting (intent detection) 
# ---------------------------------------------------------------------

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, initializers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# ----------------------- Configurable constants -----------------------
UTTER_COL = "utterance"
INTENT_COL = "intent"
SPLIT_COL = "split"              # optional: train|test
BOT_JSON = None                  # optional: path to JSON: {"topic":[ "kw1","kw2", ... ], ...}
OUTDIR = "topic_spotting_runs"   # outputs (model/metrics/label_encoder) saved here

EMBED_DIM = 300                  # loading GloVe, keep to 300; otherwise random-init
MAX_LEN = 48                     # truncate/pad length
LSTM_UNITS = 256
ATTN_HEADS = 4
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
PATIENCE = 5

# ----------------------- Utilities -----------------------

def set_tf_deterministic(seed: int = 42):
    tf.random.set_seed(seed)

def pad_sequences(seqs: List[List[int]], maxlen: int, pad_value: int = 0) -> np.ndarray:
    arr = np.full((len(seqs), maxlen), pad_value, dtype=np.int32)
    for i, s in enumerate(seqs):
        trunc = s[:maxlen]
        arr[i, :len(trunc)] = trunc
    return arr

def compute_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return num / den

# ----------------------- Tokenizer & Embeddings -----------------------

class SimpleTokenizer:
    """
    Minimal whitespace tokenizer with frequency thresholding.
    Index 0 is PAD, 1 is OOV.
    """
    def __init__(self, oov_token: str = "<UNK>"):
        self.word2idx = {oov_token: 1}
        self.idx2word = {1: oov_token}
        self.oov = oov_token

    def fit(self, texts: List[str], min_freq: int = 1):
        from collections import Counter
        c = Counter()
        for t in texts:
            for w in t.lower().split():
                c[w] += 1
        next_id = 2
        for w, f in c.items():
            if f >= min_freq and w not in self.word2idx:
                self.word2idx[w] = next_id
                self.idx2word[next_id] = w
                next_id += 1

    def encode(self, text: str) -> List[int]:
        return [self.word2idx.get(w, 1) for w in text.lower().split()]

    def vocab_size(self) -> int:
        return (max(self.word2idx.values()) if self.word2idx else 1) + 1

def load_glove_embeddings(path_txt: str, tokenizer: SimpleTokenizer, embed_dim: int) -> np.ndarray:
    """
    Load GloVe from a .txt (e.g., glove.840B.300d.txt or glove.6B.300d.txt).
    Any OOV gets random init. PAD=0 vector.
    """
    vocab_size = tokenizer.vocab_size()
    emb = np.random.uniform(-0.05, 0.05, size=(vocab_size, embed_dim)).astype("float32")
    emb[0] = 0.0  # PAD

    found = 0
    with open(path_txt, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            w, vec = parts[0], parts[1:]
            if len(vec) != embed_dim:
                continue
            if w in tokenizer.word2idx:
                emb[tokenizer.word2idx[w]] = np.asarray(vec, dtype="float32")
                found += 1
    print(f"[emb] Loaded {found} pretrained vectors / vocab_size={vocab_size}")
    return emb

def build_embedding_matrix(tokenizer: SimpleTokenizer, embed_dim: int = EMBED_DIM) -> np.ndarray:
    """Random embeddings (use load_glove_embeddings if you have a GloVe file)."""
    vocab_size = tokenizer.vocab_size()
    emb = np.random.uniform(-0.05, 0.05, size=(vocab_size, embed_dim)).astype("float32")
    emb[0] = 0.0
    return emb

# ----------------------- Self-Attention Layer -----------------------

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, num_heads: int, proj_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.proj_dim = proj_dim

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W1 = self.add_weight(shape=(d, self.proj_dim),
                                  initializer=initializers.GlorotUniform(),
                                  name="W1")
        self.W2 = self.add_weight(shape=(self.proj_dim, self.num_heads),
                                  initializer=initializers.GlorotUniform(),
                                  name="W2")
        super().build(input_shape)

    def call(self, H, mask=None):
        # H: [B, T, D], mask: [B, T]
        proj = tf.tanh(tf.matmul(H, self.W1))              # [B, T, P]
        score = tf.matmul(proj, self.W2)                   # [B, T, heads]
        if mask is not None:
            mask_exp = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            score = score + (1.0 - mask_exp) * (-1e9)
        alpha = tf.nn.softmax(score, axis=1)               # [B, T, heads]
        AHT = tf.einsum("bth,btd->bhd", alpha, H)          # [B, heads, D]
        return tf.reshape(AHT, [tf.shape(H)[0], -1])       # [B, heads*D]

# ----------------------- Model -----------------------

def build_bilstm_attention_model(vocab_size: int,
                                 embed_matrix: np.ndarray,
                                 num_classes: int,
                                 max_len: int,
                                 lstm_units: int = LSTM_UNITS,
                                 num_heads: int = ATTN_HEADS) -> tf.keras.Model:
    inp = layers.Input(shape=(max_len,), name="tokens")
    mask = tf.cast(tf.not_equal(inp, 0), tf.float32)

    emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_matrix.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(embed_matrix),
        trainable=True,
        mask_zero=True,
        name="embedding"
    )(inp)

    bilstm = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm"
    )(emb)

    att = MultiHeadSelfAttention(num_heads=num_heads, proj_dim=128, name="self_attention")(bilstm, mask=mask)

    x = layers.Dropout(0.5)(att)
    x = layers.Dense(256, activation="relu", name="dense_repr")(x)
    out = layers.Dense(num_classes, activation="softmax", name="intent")(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ----------------------- Data loading -----------------------

def load_csv_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if UTTER_COL not in df.columns or INTENT_COL not in df.columns:
        raise ValueError(f"CSV must contain columns '{UTTER_COL}' and '{INTENT_COL}'. Found: {df.columns.tolist()}")
    # Drop NA & trim
    df = df.dropna(subset=[UTTER_COL, INTENT_COL]).copy()
    df[UTTER_COL] = df[UTTER_COL].astype(str).str.strip()
    df[INTENT_COL] = df[INTENT_COL].astype(str).str.strip()
    return df

def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if SPLIT_COL in df.columns:
        train_df = df[df[SPLIT_COL].str.lower() == "train"].copy()
        val_df   = df[df[SPLIT_COL].str.lower() == "val"].copy()
        test_df  = df[df[SPLIT_COL].str.lower() == "test"].copy()
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError("Provided 'split' column must contain 'train', 'val', and 'test'.")
        return train_df, val_df, test_df
    # Else: make an 80/10/10 stratified split
    tr_df, tmp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[INTENT_COL])
    val_df, ts_df = train_test_split(tmp_df, test_size=0.5, random_state=SEED, stratify=tmp_df[INTENT_COL])
    return tr_df, val_df, ts_df

# ----------------------- BoT utilities (optional) -----------------------

def load_bag_of_topics_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_bot_embeddings(tokenizer: SimpleTokenizer,
                         embed_matrix: np.ndarray,
                         bag_of_topics: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    topic_vecs = {}
    for topic, kws in bag_of_topics.items():
        idxs = [tokenizer.word2idx.get(k.lower(), 0) for k in kws]
        vecs = embed_matrix[idxs] if len(idxs) > 0 else np.zeros((1, embed_matrix.shape[1]), dtype="float32")
        topic_vecs[topic] = vecs.mean(axis=0)
    return topic_vecs

def topic_recommendation(utt_emb: np.ndarray,
                         bot_embeddings: Dict[str, np.ndarray]) -> str:
    best_topic, best_score = None, -1.0
    for t, v in bot_embeddings.items():
        s = compute_cosine(utt_emb, v)
        if s > best_score:
            best_topic, best_score = t, s
    return best_topic

def sentence_embedding_from_model(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    penultimate = model.get_layer("dense_repr").output
    emb_model = tf.keras.Model(inputs=model.input, outputs=penultimate)
    return emb_model.predict(X, batch_size=64, verbose=0)

# ----------------------- Train / Evaluate -----------------------

def train_and_eval(model: tf.keras.Model,
                   X_train, y_train, X_val, y_val, X_test, y_test,
                   outdir: str):
    es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(os.path.join(outdir, "best_model.h5"),
                                     monitor="val_loss", save_best_only=True, save_weights_only=True)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, ckpt],
        verbose=2
    )

    probs = model.predict(X_test, batch_size=64, verbose=0)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    print("\n=== Test Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print(classification_report(y_test, preds))

    # Save metrics
    metrics_path = os.path.join(outdir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(dict(accuracy=acc, precision=p, recall=r, f1=f1), f, indent=2)

    # Save full report
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    pd.DataFrame(report).to_csv(os.path.join(outdir, "classification_report.csv"))

    return preds, probs

# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with utterance,intent[,split]")
    parser.add_argument("--glove", default=None, help="Optional path to GloVe .txt (e.g., glove.6B.300d.txt)")
    parser.add_argument("--bot", default=BOT_JSON, help="Optional path to Bag-of-Topics JSON")
    parser.add_argument("--outdir", default=OUTDIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_tf_deterministic(SEED)

    # 1) Load data
    df = load_csv_dataset(args.csv)
    train_df, val_df, test_df = split_dataframe(df)

    # 2) Tokenize
    tokenizer = SimpleTokenizer()
    tokenizer.fit(train_df[UTTER_COL].tolist())
    # Save vocab for reference
    with open(os.path.join(args.outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer.word2idx, f, ensure_ascii=False, indent=2)

    # Encode texts
    X_train_ids = [tokenizer.encode(t) for t in train_df[UTTER_COL].tolist()]
    X_val_ids   = [tokenizer.encode(t) for t in val_df[UTTER_COL].tolist()]
    X_test_ids  = [tokenizer.encode(t) for t in test_df[UTTER_COL].tolist()]

    max_len = min(MAX_LEN, max(len(s) for s in X_train_ids + X_val_ids + X_test_ids))
    X_train = pad_sequences(X_train_ids, maxlen=max_len, pad_value=0)
    X_val   = pad_sequences(X_val_ids,   maxlen=max_len, pad_value=0)
    X_test  = pad_sequences(X_test_ids,  maxlen=max_len, pad_value=0)

    # Labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[INTENT_COL].tolist())
    y_val   = le.transform(val_df[INTENT_COL].tolist())
    y_test  = le.transform(test_df[INTENT_COL].tolist())
    num_classes = len(le.classes_)
    pd.Series(le.classes_).to_csv(os.path.join(args.outdir, "label_classes.csv"), index=False, header=["class"])

    # 3) Embeddings
    if args.glove:
        embed_matrix = load_glove_embeddings(args.glove, tokenizer, EMBED_DIM)
    else:
        embed_matrix = build_embedding_matrix(tokenizer, EMBED_DIM)

    # 4) Build model
    model = build_bilstm_attention_model(
        vocab_size=tokenizer.vocab_size(),
        embed_matrix=embed_matrix,
        num_classes=num_classes,
        max_len=max_len,
        lstm_units=LSTM_UNITS,
        num_heads=ATTN_HEADS
    )
    model.summary()
    with open(os.path.join(args.outdir, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # 5) Train & Evaluate
    preds, probs = train_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, args.outdir)

    # Save model weights
    model.save_weights(os.path.join(args.outdir, "final_weights.h5"))

    # 6) Optional: topic recommendation via Bag-of-Topics
    if args.bot:
        print("\n[BoT] Loading Bag-of-Topics for topic recommendation...")
        bot = load_bag_of_topics_json(args.bot)           # {"topic": ["kw1","kw2",...]}
        bot_vecs = build_bot_embeddings(tokenizer, embed_matrix, bot)
        test_embs = sentence_embedding_from_model(model, X_test)

        recs = []
        for i in range(len(test_df)):
            utt = test_df.iloc[i][UTTER_COL]
            gold = test_df.iloc[i][INTENT_COL]
            pred_cls = le.inverse_transform([preds[i]])[0]
            rec_topic = topic_recommendation(test_embs[i], bot_vecs)
            recs.append((utt, gold, pred_cls, rec_topic))

        rec_df = pd.DataFrame(recs, columns=["utterance", "gold_intent", "predicted_intent", "recommended_topic"])
        rec_df.to_csv(os.path.join(args.outdir, "bot_recommendations.csv"), index=False, encoding="utf-8")
        print(f"[BoT] Wrote recommendations -> {os.path.join(args.outdir, 'bot_recommendations.csv')}")

    print("\nDone.")

if __name__ == "__main__":
    main()
