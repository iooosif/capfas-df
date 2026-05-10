"""
capfas_df.py — CAPaFS-DF: L-cluster + blockchain v2

Detector: cosine similarity (word + char TF-IDF) + stylometry + length

"""

import os, sys, re, json, hashlib, time
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from dataclasses import dataclass, asdict, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


from blockchain_v2 import CAPaFSBlockchain, IndependentVerifier

# 0. configuration

CFG = {
    "employee_id":   "CFO_001",
    "normal_csv":    "cfo_messages.csv",
    "test_csv":      "test_suspicious.csv",

    "word_features": 1000,
    "word_ngram":    (1, 3),
    "char_features": 800,
    "char_ngram":    (3, 5),

    # weights of chanels (sum = 1.0)
    "w_word":   0.25,
    "w_char":   0.40,
    "w_stylo":  0.20,
    "w_length": 0.15,

    # base threshold percentile — automatical calculation for huge σ
    "threshold_percentile": 5,

    "model_file":     "capfas_model.joblib",
    "results_normal": "results_normal.csv",
    "results_test":   "results_test.csv",
}

STOPWORDS = {
    "the","a","an","is","in","it","of","to","and","or","for","on","at","by",
    "as","with","this","that","be","are","was","were","have","has","had",
    "will","would","could","should","may","might","can","do","does","did",
    "not","but","from","we","i","you","he","she","they","our","your","my",
    "their","its","one","so","if","no","up","out","about","into","than",
    "then","them","there","been","also","more","any","all","just","when",
    "which","who","re","cc","fw","fwd","hi","hello","thanks","thank",
    "regards","best","sincerely","dear","pm","am","ect","hou","enron",
}



# 1. PREPROCESSING
def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower()
    t = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', t)
    t = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', ' DATE ', t)
    t = re.sub(r'\$[\d,.]+[mk]?', ' MONEY ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    tokens = [w for w in t.split() if len(w) > 2 and w not in STOPWORDS]
    return " ".join(tokens)

def clean_char(text: str) -> str:
    if not isinstance(text, str): return ""
    t = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text.lower())
    return re.sub(r'\s+', ' ', t).strip()[:2000]



# 2. Stylometry
def stylometric_features(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(10)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 2]
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)
    n_sent  = max(len(sentences), 1)
    return np.array([
        np.mean([len(w) for w in words]) if words else 0,   # avg_word_len
        n_words / n_sent,                                    # avg_sent_len
        len(set(w.lower() for w in words)) / n_words,       # vocab_richness
        len(re.findall(r'[.,;:!?"\']', text)) / n_chars,    # punct_density
        sum(1 for s in sentences if '?' in s) / n_sent,     # question_rate
        sum(1 for s in sentences if '!' in s) / n_sent,     # exclaim_rate
        len(re.findall(r',', text)) / n_words * 100,        # comma_rate
        sum(1 for c in text if c.isupper()) / n_chars,      # upper_rate
        n_sent / max(len(text.split('\n\n')), 1),            # avg_para_len
        sum(1 for c in text if c.isdigit()) / n_chars,      # digit_density
    ], dtype=float)

def message_length_features(text: str) -> np.ndarray:
    if not isinstance(text, str): return np.zeros(3)
    return np.array([len(text), len(text.split()),
                     max(len(re.split(r'[.!?]+', text)), 1)], dtype=float)



# 3. L-cluster

@dataclass
class ChannelStats:
    mean: float = 0.0
    std:  float = 0.0
    threshold: float = 0.0
    fp_rate: float = 0.0


class LClusterProfile:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.vec_word:      TfidfVectorizer = None
        self.vec_char:      TfidfVectorizer = None
        self.centroid_word: np.ndarray = None
        self.centroid_char: np.ndarray = None
        self.stylo_mean:    np.ndarray = None
        self.stylo_std:     np.ndarray = None
        self.len_mean:      np.ndarray = None
        self.len_std:       np.ndarray = None
        self.stats = ChannelStats()
        self.profile_bytes: bytes = None

    # training
    def fit(self, texts: list[str]):
        texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 10]
        n = len(texts)
        print(f"\n[L-Cluster] Training profile on {n} documents...")

        # Word TF-IDF
        word_clean = [clean_text(t) for t in texts]
        self.vec_word = TfidfVectorizer(
            ngram_range=self.cfg["word_ngram"], max_features=self.cfg["word_features"],
            sublinear_tf=True, min_df=2, analyzer="word")
        X_word = normalize(self.vec_word.fit_transform(word_clean), norm="l2")
        self.centroid_word = np.asarray(X_word.mean(axis=0))
        sim_word = cosine_similarity(X_word, self.centroid_word).flatten()

        # Char TF-IDF
        char_clean = [clean_char(t) for t in texts]
        self.vec_char = TfidfVectorizer(
            ngram_range=self.cfg["char_ngram"], max_features=self.cfg["char_features"],
            sublinear_tf=True, min_df=2, analyzer="char_wb")
        X_char = normalize(self.vec_char.fit_transform(char_clean), norm="l2")
        self.centroid_char = np.asarray(X_char.mean(axis=0))
        sim_char = cosine_similarity(X_char, self.centroid_char).flatten()

        # Stylometry
        stylo_matrix = np.array([stylometric_features(t) for t in texts])
        self.stylo_mean = stylo_matrix.mean(axis=0)
        self.stylo_std  = np.where(stylo_matrix.std(axis=0) < 1e-9, 1.0, stylo_matrix.std(axis=0))
        sim_stylo = 1.0 / (1.0 + np.abs((stylo_matrix - self.stylo_mean) / self.stylo_std).mean(axis=1))

        # Length
        len_matrix = np.array([message_length_features(t) for t in texts])
        self.len_mean = len_matrix.mean(axis=0)
        self.len_std  = np.where(len_matrix.std(axis=0) < 1e-9, 1.0, len_matrix.std(axis=0))
        sim_len = 1.0 / (1.0 + np.abs((len_matrix - self.len_mean) / self.len_std).mean(axis=1))

        # Final score
        w = self.cfg
        final = (w["w_word"] * sim_word + w["w_char"] * sim_char +
                 w["w_stylo"] * sim_stylo + w["w_length"] * sim_len)

        mu  = float(final.mean())
        std = float(final.std())

        base = self.cfg.get("threshold_percentile", 5)
        if std > 0.12:
            pct = base + 20
        elif std > 0.08:
            pct = base + 10
        else:
            pct = base
        thr = round(float(np.percentile(final, pct)), 6)
        fp  = float((final < thr).mean())

        self.stats = ChannelStats(mean=mu, std=std, threshold=thr, fp_rate=fp)

        self.profile_bytes = json.dumps({
            "centroid_word_sum": float(self.centroid_word.sum()),
            "centroid_char_sum": float(self.centroid_char.sum()),
            "stylo_mean": self.stylo_mean.tolist(),
            "len_mean": self.len_mean.tolist(),
            "threshold": thr, "mu": mu, "std": std,
        }, sort_keys=True).encode()

        return {
            "n_train": n,
            "word_vocab": len(self.vec_word.vocabulary_),
            "char_vocab": len(self.vec_char.vocabulary_),
            "mu": round(mu, 4), "std": round(std, 4),
            "threshold": thr, "fp_rate": fp, "percentile": pct,
            "channel_means": {
                "word":  round(float(sim_word.mean()), 4),
                "char":  round(float(sim_char.mean()), 4),
                "stylo": round(float(sim_stylo.mean()), 4),
                "len":   round(float(sim_len.mean()), 4),
            }
        }, {"word": sim_word, "char": sim_char,
            "stylo": sim_stylo, "len": sim_len, "final": final}

    # scoring
    def score(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {"word": 0.0, "char": 0.0, "stylo": 0.0, "len": 0.0, "final": 0.0}

        w_vec  = normalize(self.vec_word.transform([clean_text(text)]), norm="l2")
        s_word = float(cosine_similarity(w_vec, self.centroid_word)[0][0])

        c_vec  = normalize(self.vec_char.transform([clean_char(text)]), norm="l2")
        s_char = float(cosine_similarity(c_vec, self.centroid_char)[0][0])

        sf      = stylometric_features(text)
        s_stylo = float(1.0 / (1.0 + np.abs((sf - self.stylo_mean) / self.stylo_std).mean()))

        lf    = message_length_features(text)
        s_len = float(1.0 / (1.0 + np.abs((lf - self.len_mean) / self.len_std).mean()))

        w = self.cfg
        final = (w["w_word"] * s_word + w["w_char"] * s_char +
                 w["w_stylo"] * s_stylo + w["w_length"] * s_len)

        return {"word": round(s_word, 4), "char": round(s_char, 4),
                "stylo": round(s_stylo, 4), "len": round(s_len, 4),
                "final": round(final, 4)}

    def classify(self, text: str) -> tuple[str, dict, str]:
        scores = self.score(text)
        thr = self.stats.threshold
        if scores["final"] >= thr:
            return "NORMAL", scores, f"match (final={scores['final']:.4f} ≥ thr={thr:.4f})"
        else:
            return "ANOMALY", scores, (
                f"ANOMALY: final={scores['final']:.4f} < thr={thr:.4f} "
                f"[w={scores['word']:.3f} c={scores['char']:.3f} "
                f"s={scores['stylo']:.3f} l={scores['len']:.3f}]")

    #lexical portrait
    def lexical_portrait(self, texts: list[str]) -> dict:
        all_words, all_bi, all_tri, stylo_all, len_all = [], [], [], [], []
        for text in texts:
            tokens = clean_text(text).split()
            all_words.extend(tokens)
            all_bi.extend(" ".join(tokens[i:i+2]) for i in range(len(tokens)-1))
            all_tri.extend(" ".join(tokens[i:i+3]) for i in range(len(tokens)-2))
            stylo_all.append(stylometric_features(text))
            len_all.append(message_length_features(text))
        stylo_arr = np.array(stylo_all)
        len_arr   = np.array(len_all)
        stylo_labels = ["avg_word_len","avg_sent_len","vocab_richness","punct_density",
                        "question_rate","exclaim_rate","comma_rate","upper_rate",
                        "avg_para_len","digit_density"]
        return {
            "top_words":    Counter(all_words).most_common(30),
            "top_bigrams":  Counter(all_bi).most_common(20),
            "top_trigrams": Counter(all_tri).most_common(15),
            "stylometry": {
                lbl: {"mean": round(float(stylo_arr[:, i].mean()), 4),
                      "std":  round(float(stylo_arr[:, i].std()), 4)}
                for i, lbl in enumerate(stylo_labels)
            },
            "message_length": {
                lbl: {"mean": round(float(len_arr[:, i].mean()), 1),
                      "std":  round(float(len_arr[:, i].std()), 1)}
                for i, lbl in enumerate(["n_chars","n_words","n_sents"])
            },
        }

    # saving/loading
    def save(self):
        joblib.dump({
            "vec_word": self.vec_word, "vec_char": self.vec_char,
            "centroid_word": self.centroid_word, "centroid_char": self.centroid_char,
            "stylo_mean": self.stylo_mean, "stylo_std": self.stylo_std,
            "len_mean": self.len_mean, "len_std": self.len_std,
            "stats": asdict(self.stats), "cfg": self.cfg,
        }, self.cfg["model_file"])
        print(f"[L-Cluster] Model saved: {self.cfg['model_file']}")

    def load(self) -> bool:
        if not os.path.exists(self.cfg["model_file"]): return False
        d = joblib.load(self.cfg["model_file"])
        self.vec_word = d["vec_word"]; self.vec_char = d["vec_char"]
        self.centroid_word = d["centroid_word"]; self.centroid_char = d["centroid_char"]
        self.stylo_mean = d["stylo_mean"]; self.stylo_std = d["stylo_std"]
        self.len_mean = d["len_mean"]; self.len_std = d["len_std"]
        self.stats = ChannelStats(**d["stats"])
        return True



# 4 helper functions for blockchain interaction and reporting

def sep(title=""):
    print("\n" + "="*65 + (f"\n  {title}\n" + "="*65 if title else ""))


def print_last_blocks(bc: CAPaFSBlockchain, n: int = 5):
    """Print the last n blocks of the chain."""
    print(f"\n--- Last {n} blocks ---")
    for block in bc.chain[-n:]:
        d = block.data
        extra = ""
        if d.similarity_score is not None:
            extra = f" sim={d.similarity_score:.4f}"
        if d.metadata.get("severity"):
            extra += f" [{d.metadata['severity']}]"
        print(f"  #{block.index:03d} | {d.event_type:<22} | "
              f"hash: {block.hash[:12]}... | "
              f"prev: {block.previous_hash[:12]}...{extra}")



# 5. PIPELINE
def _read_file_bytes(path: str) -> bytes:
    """Reads a file and returns bytes for hashing."""
    with open(path, "rb") as f:
        return f.read()


def run_pipeline(force_retrain: bool = False):
    sep("CAPaFS-DF v2: L-cluster+ blockchain")

    if not os.path.exists(CFG["normal_csv"]):
        print(f"[!] {CFG['normal_csv']} not found. Please run enron_loader.py")
        sys.exit(1)

    df_normal = pd.read_csv(CFG["normal_csv"]).dropna(subset=["text"])
    df_test   = pd.read_csv(CFG["test_csv"]).dropna(subset=["text"])
    print(f"[✓] Loaded: {len(df_normal)} normal | {len(df_test)} test")

    #  BLOCKCHAIN v2: loading existing chain (not deleting!) 
    bc = CAPaFSBlockchain(
        employee_id=CFG["employee_id"],
        load_existing=True,
        enable_smart_contract=True,
    )
    model_file    = CFG["model_file"]
    chain_has_profile = bc.get_current_profile_hash() is not None
    model_exists      = os.path.exists(model_file)

    # selecting mode
    # TRAIN 
    # VERIFYь
    do_train = force_retrain or not model_exists or not chain_has_profile
    mode = "train (first run)" if (not chain_has_profile) else \
           "retrain (force_retrain)" if force_retrain else \
           "verify (re-run)"
    print(f"[BC] Mode: {mode}  |  Blocks in chain: {len(bc.chain)}")
    

    profile = LClusterProfile(CFG)

    if do_train:
        #TRAINING PATH
        sep("TRAINING PROFILE")
        stats, _ = profile.fit(df_normal["text"].tolist())

        print(f"\n  Documents           : {stats['n_train']}")
        print(f"  σ final score       : {stats['std']:.4f}  {'(heterogeneous dataset)' if stats['std'] > 0.08 else '(homogeneous)'}")
        print(f"  Adaptive percentile  : {stats['percentile']}")
        print(f"  Threshold           : {stats['threshold']:.4f}")
        print(f"  FP on training      : {stats['fp_rate']*100:.1f}%")
        print(f"\n  Mean by channel:")
        for k, v in stats["channel_means"].items():
            print(f"    {k:<8}: {v:.4f}")

        # Saving of the model on disk — HASHING THE FILE, not centroids
        profile.save()
        file_bytes = _read_file_bytes(model_file)

        model_params = {
            "threshold": stats["threshold"],
            "n_train":   stats["n_train"],
            "sigma":     stats["std"],
            "percentile": stats["percentile"],
        }
        if not chain_has_profile:
            bc.register_profile(file_bytes, model_params)
            print(f"[BC] Profile (file hash) registered → block #{len(bc.chain)-1}")
        else:
            bc.update_profile(file_bytes, reason="force_retrain")
            print(f"[BC] Profile (file hash) updated → block #{len(bc.chain)-1}")
        

    else:
        #  VERIFICATION PATH 
        # Read the model file from disk as is — hash the binary bytes
        # if someone change file, hash will not match blockchain
        sep("VERIFICATION PROFILE (BLOCKCHAIN GATE)")

        try:
            file_bytes = _read_file_bytes(model_file)
        except Exception as e:
            print(f"[BC] ✗ We failed to read the model file: {e}")
            print(f"[BC]   Classification BLOCKED.")
            sys.exit(2)

        ok, reason = bc.verify_profile_against_chain(file_bytes)
        if ok:
            print(f"[BC] ✓ File hash matches blockchain: {reason}")
            print(f"[BC]   Hash in chain   : {bc.get_current_profile_hash()[:32]}...")
            print(f"[BC]   Classification ALLOWED")
        else:
            print(f"[BC] ✗ VERIFICATION FAILED: {reason}")
            print(f"[BC]   File '{model_file}' does not match registered profile.")
            print(f"[BC]   Block TAMPER_DETECTED recorded in chain.")
            print(f"[BC]   Classification BLOCKED — model may have been tampered with.")
            bc.print_summary()
            sys.exit(2)

        # Loading model from file (without retraining)
        if not profile.load():
            print(f"[!] We failed to load the model from {model_file}. Run without --verify.")
            sys.exit(1)
        print(f"[BC]   Model loaded from {model_file} (without retraining)")


    # Lexical portrait
    sep("LEXICAL PORTRAIT OF EMPLOYEE")
    portrait = profile.lexical_portrait(df_normal["text"].tolist())
    print("\n▸ top 30 words:")
    wdf = pd.DataFrame(portrait["top_words"], columns=["Word", "Freq"])
    print(wdf.to_string(index=False))
    print("\n▸ top 20 bigrams:")
    bdf = pd.DataFrame(portrait["top_bigrams"], columns=["Bigram", "Freq"])
    print(bdf.to_string(index=False))
    print("\n▸ top 15 trigrams:")
    tdf = pd.DataFrame(portrait["top_trigrams"], columns=["Trigram", "Freq"])
    print(tdf.to_string(index=False))
    print("\n▸ STYLOMETRY:")
    for feat, vals in portrait["stylometry"].items():
        print(f"  {feat:<20}: µ={vals['mean']:.4f}  σ={vals['std']:.4f}")
    print("\n▸ MESSAGE LENGTH:")
    for feat, vals in portrait["message_length"].items():
        print(f"  {feat:<20}: µ={vals['mean']:.1f}  σ={vals['std']:.1f}")

    wdf.to_csv("portrait_words.csv", index=False)
    bdf.to_csv("portrait_bigrams.csv", index=False)
    tdf.to_csv("portrait_trigrams.csv", index=False)
    pd.DataFrame([{"feature": k, **v}
                  for k, v in portrait["stylometry"].items()]).to_csv("portrait_stylometry.csv", index=False)

    # Classification
    sep("CLASSIFICATION")

    def classify_df(df):
        rows = []
        for text in df["text"]:
            lbl, scores, reason = profile.classify(str(text))
            rows.append({"text": text, "label": lbl, "reason": reason, **scores})
        return pd.DataFrame(rows)

    res_normal = classify_df(df_normal)
    res_test   = classify_df(df_test)

    # When severity HIGH/CRITICAL automatically record an incident in blockchain with details (text, similarity, reason).
    # TRANSACTION_BLOCKED and SECURITY_ALERT 
    anomalies = res_test[res_test["label"] == "ANOMALY"]
    for _, row in anomalies.iterrows():
        bc.log_incident(
            text=str(row["text"]),
            similarity=row["final"],
            threshold=profile.stats.threshold,
        )
    print(f"\n[BC] incident writing: {len(anomalies)}")
    print(f"[BC] Blocks in chain after classification: {len(bc.chain)}")
  

    # Detection metrics
    sep("DETECTION METRICS")
    fp   = len(res_normal[res_normal["label"] == "ANOMALY"])
    tp   = len(res_test[res_test["label"]     == "ANOMALY"])
    tn   = len(res_normal) - fp
    fn   = len(res_test)   - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0

    print(f"\n  TP (attack detected)   : {tp} / {len(res_test)}  ({rec*100:.1f}%)")
    print(f"  FN (missed)            : {fn}")
    print(f"  FP (false alarms)      : {fp} / {len(res_normal)}  ({fp/len(res_normal)*100:.2f}%)")
    print(f"  TN                     : {tn}")
    print(f"\n  Precision              : {prec:.4f}")
    print(f"  Recall                 : {rec:.4f}")
    print(f"  F1-score               : {f1:.4f}")
    print(f"\n  Scores normal: µ={res_normal['final'].mean():.4f}  σ={res_normal['final'].std():.4f}")
    print(f"  Scores test:   µ={res_test['final'].mean():.4f}  σ={res_test['final'].std():.4f}")
    print(f"  Gap:              {res_normal['final'].mean() - res_test['final'].mean():.4f}")
    print(f"\n  By channels:")
    for ch in ["word","char","stylo","len"]:
        mn, mt = res_normal[ch].mean(), res_test[ch].mean()
        print(f"    {ch:<8}: normal={mn:.4f}  test={mt:.4f}  gap={mn-mt:.4f}")

    
    bc.print_summary()
    print_last_blocks(bc, n=6)

    # Independent verification (the role of the regulator)
    sep("INDEPENDENT VERIFICATION (THE ROLE OF THE REGULATOR)")
    verifier_result = IndependentVerifier().verify()
    print(f"  Chain integrity      : {'✓ OK' if verifier_result['ok'] else '✗ BROKEN'}")
    print(f"  Blocks verified      : {verifier_result['blocks']}")
    print(f"  Anchor consistency   : {verifier_result['anchor_consistent']}")
    print(f"  Verifier             : {verifier_result['verifier']}")
   
    # Result saving
    res_normal.to_csv(CFG["results_normal"], index=False)
    res_test.to_csv(CFG["results_test"], index=False)
    pd.DataFrame([{"metric": k, "value": v} for k, v in {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": round(prec, 4), "Recall": round(rec, 4),
        "F1": round(f1, 4), "Threshold": profile.stats.threshold,
        "Gap": round(res_normal['final'].mean() - res_test['final'].mean(), 4),
    }.items()]).to_csv("capfas_metrics.csv", index=False)

    # Blockchain metrics in a separate file
    bc_summary = bc.summary()
    pd.DataFrame([{"metric": k, "value": str(v)}
                  for k, v in bc_summary.items()]).to_csv("blockchain_metrics.csv", index=False)

    sep("DONE")
    print("  results_normal.csv, results_test.csv, capfas_metrics.csv")
    print("  portrait_*.csv, capfas_chain.json, blockchain_metrics.csv")
    print(f"  Chain contains {len(bc.chain)} blocks (accumulated over all runs)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain the model and update the profile in the blockchain")
    args = parser.parse_args()
    run_pipeline(force_retrain=args.retrain)