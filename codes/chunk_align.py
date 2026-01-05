import json, re, numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from pathlib import Path
from codes.util import Embedder

ALL_LABELS = [
    "Title","Section-header","Text","List-item","Formula",
    "Picture","Table","Caption","Footnote","Page-header","Page-footer"
    ]
MISMATCH_LABELS = [
    "Title","Section-header","Text","List-item","Formula",
    "Picture","Table","Caption","Footnote","Page-header","Page-footer"
    ]
CHUNK_MARK = "<<<CHUNK>>>"

_LABEL2IDX = {lab: i for i, lab in enumerate(ALL_LABELS)}
_MISMATCH_IDXS = [_LABEL2IDX[lab] for lab in MISMATCH_LABELS]

# utils
def read_chunks(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]

def read_chunks_from_txt_file(path: str) -> List[Dict]:
    chunks = []
    cur = {lab: "" for lab in ALL_LABELS}

    def flush():
        if any(v.strip() for v in cur.values()):
            chunks.append({
                "labels": {lab: {"text": cur[lab].strip()} for lab in ALL_LABELS}
            })
        for lab in ALL_LABELS:
            cur[lab] = ""

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("<<<CHUNK>>>"):
                flush()
                continue

            if line.startswith("<") and ">" in line:
                lab, text = line[1:].split(">", 1)
                if lab in cur:
                    cur[lab] += text.strip() + " "

    flush()
    return chunks

def read_pairs(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_chunks_txt(p: Path) -> List[str]:
    raw = p.read_text(encoding="utf-8")
    parts = [s.strip() for s in raw.split(CHUNK_MARK)]
    return [s for s in (re.sub(r"[ \t]+"," ", x.replace("\u3000"," ")).strip() for x in parts) if s]

def cosine_sim(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def sigmoid(x): return 1/(1+np.exp(-x))

def build_per_label_texts(chunks: List[Dict]) -> List[List[str]]:
    texts_per_chunk = []
    for ch in chunks:
        texts = []
        for lab in ALL_LABELS:
            texts.append(ch["labels"].get(lab, {}).get("text",""))
        texts_per_chunk.append(texts)
    return texts_per_chunk

def mismatch_feats_from_emb(EMB_src, EMB_tgt, i, j) -> List[float]:
    # 0/1: XOR of existence (only one side has that label embedding)
    out = []
    for li in _MISMATCH_IDXS:
        s = (np.linalg.norm(EMB_src[i, li]) > 0)
        t = (np.linalg.norm(EMB_tgt[j, li]) > 0)
        out.append(1.0 if (s != t) else 0.0)
    return out

def build_feats(EMB_src, EMB_tgt, i, j) -> List[float]:
    sims = [cosine_sim(EMB_src[i, li], EMB_tgt[j, li]) for li in range(len(ALL_LABELS))]
    mm = mismatch_feats_from_emb(EMB_src, EMB_tgt, i, j)     # 0/1
    return sims + [-v for v in mm]                           # fired => -1, else 0

# for train mode
def read_matches(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_gold_map(matches: List[Dict], langs) -> Dict[int, set]:
    src_key = f"{langs[0]}_idx"  # e.g., "en_idx"
    tgt_key = f"{langs[1]}_idx"  # e.g., "ja_idx"
    gold = {}
    for row in matches:
        i = int(row.get(src_key, row.get("en_idx", -1)))
        if i < 0:
            continue
        js = set()
        for m in (row.get("matches") or []):
            if tgt_key in m:
                js.add(int(m[tgt_key]))
            elif "ja_idx" in m:
                js.add(int(m["ja_idx"]))
        gold[i] = js
    return gold

def build_train_pairs_from_gold(
        gold_map: Dict[int, set],
        n_src: int,
        n_tgt: int,
        neg_per_pos: int = 5,
        seed: int = 0
    ) -> List[Dict]:

    rng = np.random.default_rng(seed)
    pairs = []

    for i in range(n_src):
        pos_js = sorted(list(gold_map.get(i, set())))
        for j in pos_js:
            if 0 <= j < n_tgt:
                pairs.append({"src_idx": i, "tgt_idx": j, "label": 1})

        # negatives
        if pos_js and neg_per_pos > 0:
            pos_set = set(pos_js)
            need = neg_per_pos * len(pos_js)
            picked = set()
            trials = 0
            while len(picked) < need and trials < need * 20:
                jj = int(rng.integers(0, n_tgt))
                if jj not in pos_set:
                    picked.add(jj)
                trials += 1
            for jj in picked:
                pairs.append({"src_idx": i, "tgt_idx": int(jj), "label": 0})
    return pairs

def train_weights(EMB_src, EMB_tgt, train_pairs, langs):
    X, y = [], []
    for pair in train_pairs:
        i, j, gold = pair[f"{langs[0]}_idx"], pair[f"{langs[1]}_idx"], pair["label"]
        X.append(build_feats(EMB_src, EMB_tgt, i, j))
        y.append(gold)

    clf = LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y))
    coef = clf.coef_[0]
    bias = float(clf.intercept_[0])

    n_sim = len(ALL_LABELS)
    w = {lab: float(c) for lab, c in zip(ALL_LABELS, coef[:n_sim])}
    lambdas = {lab: float(c) for lab, c in zip(MISMATCH_LABELS, coef[n_sim:])}

    # optional safety: keep penalties non-negative (simple)
    for k in lambdas:
        if lambdas[k] < 0:
            lambdas[k] = 0.0

    return w, lambdas, bias

def run_train(model_emb: str, out_path: Path, langs, neg_per_pos: int = 5, seed: int = 0):

    src_chunks_txt = "data/to_dev/gold/all_chunks_ordered_label-en.txt"
    tgt_chunks_txt = "data/to_dev/gold/all_chunks_ordered_label-ja.txt"
    src_chunks = read_chunks_from_txt_file(str(src_chunks_txt))
    tgt_chunks = read_chunks_from_txt_file(str(tgt_chunks_txt))

    src_texts = build_per_label_texts(src_chunks)
    tgt_texts = build_per_label_texts(tgt_chunks)

    embedder = Embedder(model_emb)
    EMB_src = embedder.encode_label_texts(src_texts)
    EMB_tgt = embedder.encode_label_texts(tgt_texts)

    matches = read_matches(str("data/to_dev/gold/matches.json"))
    gold_map = build_gold_map(matches, langs)

    # build binary pairs
    pairs = build_train_pairs_from_gold(
        gold_map,
        n_src=len(src_texts),
        n_tgt=len(tgt_texts),
        neg_per_pos=neg_per_pos,
        seed=seed
    )

    src_key = f"{langs[0]}_idx"
    tgt_key = f"{langs[1]}_idx"
    train_pairs = [{src_key: p["src_idx"], tgt_key: p["tgt_idx"], "label": p["label"]} for p in pairs]

    w, lambdas, bias = train_weights(EMB_src, EMB_tgt, train_pairs, langs)

    w_vec = np.array([w[lab] for lab in ALL_LABELS], dtype=np.float32)
    lam_vec = np.array([lambdas.get(lab, 0.0) for lab in MISMATCH_LABELS], dtype=np.float32)

    preds, golds = [], []
    for p in train_pairs:
        i, j, gold = int(p[src_key]), int(p[tgt_key]), int(p["label"])
        feats = build_feats(EMB_src, EMB_tgt, i, j)
        sims = np.array(feats[:len(ALL_LABELS)], dtype=np.float32)
        neg_mm = np.array(feats[len(ALL_LABELS):], dtype=np.float32)
        logit = float(bias + np.dot(w_vec, sims) + np.dot(lam_vec, neg_mm))
        score = sigmoid(logit)
        preds.append(1 if score >= 0.5 else 0)
        golds.append(gold)

    train_f1 = float(f1_score(golds, preds)) if golds else 0.0

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"weights": w, "lambdas": lambdas, "bias": bias, "train_f1": train_f1},
            f, ensure_ascii=False, indent=2
        )
    return out_path


# for align mode
def align_chunks(EMB_src, EMB_tgt, src_texts, tgt_texts, w, lambdas, bias, thresh_chunk_sim, langs):
    # fast vectors aligned to features
    w_vec = np.array([w[lab] for lab in ALL_LABELS], dtype=np.float32)
    lam_vec = np.array([lambdas.get(lab, 0.0) for lab in MISMATCH_LABELS], dtype=np.float32)

    matches = []
    for i in range(len(src_texts)):
        cand_list = []
        for j in range(len(tgt_texts)):
            feats = build_feats(EMB_src, EMB_tgt, i, j)
            sims = np.array(feats[:len(ALL_LABELS)], dtype=np.float32)
            neg_mm = np.array(feats[len(ALL_LABELS):], dtype=np.float32)

            logit = float(bias + np.dot(w_vec, sims) + np.dot(lam_vec, neg_mm))
            score = sigmoid(logit)

            if score >= thresh_chunk_sim:
                cand_list.append({f"{langs[1]}_idx": j, "score": round(float(score), 4)})

        cand_list.sort(key=lambda x: x["score"], reverse=True)
        matches.append({f"{langs[0]}_idx": i, "matches": cand_list})
    return matches

def export_pairs_from_txt(src_dir: Path, tgt_dir: Path, matches: List[Dict], out_txt_dir: Path, langs) -> None:
    src_unl = src_dir / "all_chunks_ordered.txt"  #unlabeled
    src_lbl = src_dir / "all_chunks_ordered_label.txt"  #labeled
    tgt_unl = tgt_dir / "all_chunks_ordered.txt"
    tgt_lbl = tgt_dir / "all_chunks_ordered_label.txt"
    if not (src_unl.exists() and src_lbl.exists() and tgt_unl.exists() and tgt_lbl.exists()):
        raise FileNotFoundError("ordered txt not found (unlabeled/labeled).")

    SRC_U = read_chunks_txt(src_unl)
    SRC_L = read_chunks_txt(src_lbl)
    TGT_U = read_chunks_txt(tgt_unl)
    TGT_L = read_chunks_txt(tgt_lbl)

    # bring corresponding tgt chunk texts
    for ci in range(len(SRC_U)):
        # SRC
        (out_txt_dir / f"chunk{ci+1}-{langs[0]}.txt").write_text(SRC_U[ci] if ci < len(SRC_U) else "", encoding="utf-8")
        (out_txt_dir / f"chunk{ci+1}-label-{langs[0]}.txt").write_text(SRC_L[ci] if ci < len(SRC_L) else "", encoding="utf-8")

        # TGT
        tgt_idxs = []
        if ci < len(matches):
            tgt_idxs = [int(x.get(f"{langs[1]}_idx")) for x in (matches[ci].get("matches") or []) if f"{langs[1]}_idx" in x]
        tgt_txt_u = "\n".join(TGT_U[j] for j in tgt_idxs if 0 <= j < len(TGT_U))
        tgt_txt_l = "\n".join(TGT_L[j] for j in tgt_idxs if 0 <= j < len(TGT_L))

        (out_txt_dir / f"chunk{ci+1}-{langs[1]}.txt").write_text(tgt_txt_u, encoding="utf-8")
        (out_txt_dir / f"chunk{ci+1}-label-{langs[1]}.txt").write_text(tgt_txt_l, encoding="utf-8")

def run_chunkalign(src_chunks_dir, tgt_chunks_dir, path_reg_weights, model_emb,
                thresh_chunk_sim, out_json, out_txt_dir, langs):

    out_txt_dir.mkdir(parents=True, exist_ok=True)

    src_chunks_txt = src_chunks_dir / "all_chunks_ordered_label.txt"
    tgt_chunks_txt = tgt_chunks_dir / "all_chunks_ordered_label.txt"

    src_chunks = read_chunks_from_txt_file(str(src_chunks_txt))
    tgt_chunks = read_chunks_from_txt_file(str(tgt_chunks_txt))

    src_texts = build_per_label_texts(src_chunks)
    tgt_texts = build_per_label_texts(tgt_chunks)

    embedder = Embedder(model_emb)
    EMB_src = embedder.encode_label_texts(src_texts)
    EMB_tgt = embedder.encode_label_texts(tgt_texts)

    with open(path_reg_weights, "r", encoding="utf-8") as f:
        data = json.load(f)
    w = data["weights"]
    lambdas = data["lambdas"]
    bias = float(data.get("bias", 0.0))

    matches = align_chunks(EMB_src, EMB_tgt, src_texts, tgt_texts, w, lambdas, bias, thresh_chunk_sim, langs)
    with open(str(out_json), "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    # organize texts based on the matches
    export_pairs_from_txt(src_chunks_dir, tgt_chunks_dir, matches, out_txt_dir, langs)
    return out_json