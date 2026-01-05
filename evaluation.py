from __future__ import annotations
import argparse
import datetime as dt
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from codes.util import setup_logger
from model.chunk_detection import run_chunk_detection
from model.chunk_alignment import run_chunk_alignment
from model.sentence_alignment import run_sentence_alignment

ROOT = Path(__file__).resolve().parent
DATA_PDF = ROOT / "data/to_eval/pdf"
GOLD = ROOT / "data/to_eval/gold"
RES = ROOT / "result"
CHUNK_SEP = "<<<CHUNK>>>"
WS = re.compile(r"\s+", re.UNICODE)

CONFIG_PATH = ROOT / "tools/config.json"


# =========================
# Basic text utils
# =========================
def norm(s: str) -> str:
    return WS.sub(" ", (s or "").strip())

def toks(s: str) -> List[str]:
    s = norm(s).lower()
    return re.findall(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]+", s)

def jac(a: str, b: str) -> float:
    A, B = set(toks(a)), set(toks(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def bleu(h: str, r: str) -> float:
    """Prefer sacrebleu if available; otherwise fall back to Jaccard."""
    h, r = norm(h), norm(r)
    if not h and not r:
        return 1.0
    if not h or not r:
        return 0.0
    try:
        import sacrebleu  # type: ignore

        return sacrebleu.sentence_bleu(h, [r]).score / 100.0
    except Exception:
        return jac(h, r)

def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = (2 * p * r) / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}

def f(x):  # short float
    return f"{x:.3f}"

def print_eval_summary(report: dict):
    meta = report["meta"]
    evalset = meta["evalset"]
    align_method = meta["align_method"]

    print(f"\nEvalset: {evalset}")
    print("=" * 55)

    # ---------------- Chunk Detection ----------------
    print("Chunk Detection")
    print("lang | #chunk(pred/gold) | P     R     F1    | Avg BLEU")
    print("-" * 55)

    for lang in ("en", "ja"):
        cd = report["chunk_detection"][lang]
        p = cd["boundary_strict"]["precision"]
        r = cd["boundary_strict"]["recall"]
        f1 = cd["boundary_strict"]["f1"]
        bleu = cd["avg_chunk_bleu_strict"]["avg_bleu"]
        n_calculated = int(cd["avg_chunk_bleu_strict"]["n"])
        n_pred = cd["avg_chunk_bleu_strict"]["chunks pred/gold"]

        print(f"{lang:2}   | {n_pred:<12} | "
              f"{f(p)} {f(r)} {f(f1)} | {f(bleu)} ({n_calculated} pairs)\n")

    print("=" * 55)

    # ---------------- Chunk Alignment ----------------
    if report["chunk_alignment"] is not None:
        print("Chunk Alignment")
        print("      | #pairs(pred/gold) | P     R     F1    | Avg BLEU")
        print("-" * 55)

        ca = report["chunk_alignment"]
        p = ca["target_boundary_strict"]["precision"]
        r = ca["target_boundary_strict"]["recall"]
        f1 = ca["target_boundary_strict"]["f1"]
        bleu = ca["avg_target_chunk_bleu_strict"]["avg_bleu"]
        n_calculated = int(ca["avg_target_chunk_bleu_strict"]["n"])
        n_pred = ca["avg_target_chunk_bleu_strict"]["aligned chunks pred/gold"]

        print(f"{lang:2}   | {n_pred:<12} | "
              f"{f(p)} {f(r)} {f(f1)} | {f(bleu)} ({n_calculated} pairs)\n")

        print("=" * 55)

    # ---------------- Sentence Alignment ----------------
    sa = report["sentence_alignment"]
    p = sa["strict"]["precision"]
    r = sa["strict"]["recall"]
    f1 = sa["strict"]["f1"]
    n_gold = int(sa["counts"]["gold_pairs"])
    n_pred = int(sa["counts"]["pred_pairs"])

    print(f"Sentence Alignment ({align_method}, strict)")
    print("      | #pairs(pred/gold) | P     R     F1")
    print("-" * 55)
    print(f"sent | {n_pred}/{n_gold:<12} | {f(p)} {f(r)} {f(f1)}\n")
    print("=" * 55)
    print()


# =========================
# Chunk parsing & matching
# =========================
@dataclass
class Chunk:
    boundary: str
    text: str
    lines: List[str]

def read_chunks(p: Path) -> List[Chunk]:
    raw = p.read_text(encoding="utf-8", errors="replace")
    out: List[Chunk] = []
    for part in raw.split(CHUNK_SEP):
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        b = norm(lines[0]) if lines else ""
        out.append(Chunk(b, part, lines))
    return out

def best_match_idx(boundary: str, pred: Sequence[Chunk], strict: bool, th: float) -> Optional[int]:
    b = norm(boundary)
    if not b:
        return None
    if strict:
        for i, c in enumerate(pred):
            if norm(c.boundary) == b:
                return i
        return None
    best_i, best_sc = None, -1.0
    for i, c in enumerate(pred):
        sc = jac(b, c.boundary)
        if sc > best_sc:
            best_sc, best_i = sc, i
    return best_i if best_i is not None and best_sc >= th else None

def greedy_match(gold: Sequence[str], pred: Sequence[str], strict: bool, th: float) -> Tuple[int, int, int]:
    G = [norm(x) for x in gold if norm(x)]
    P = [norm(x) for x in pred if norm(x)]
    used_g, used_p = set(), set()

    if strict:
        idx: Dict[str, List[int]] = {}
        for i, g in enumerate(G):
            idx.setdefault(g, []).append(i)
        tp = 0
        for j, p in enumerate(P):
            for i in idx.get(p, []):
                if i not in used_g:
                    used_g.add(i)
                    used_p.add(j)
                    tp += 1
                    break
        return tp, len(P) - tp, len(G) - tp

    cand = [(i, j, jac(g, p)) for i, g in enumerate(G) for j, p in enumerate(P) if jac(g, p) >= th]
    cand.sort(key=lambda x: x[2], reverse=True)
    tp = 0
    for i, j, _ in cand:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        tp += 1
    return tp, len(P) - tp, len(G) - tp


# =========================
# Chunk detection eval
# =========================
def eval_chunk_det(gold_txt: Path, pred_txt: Path, th: float) -> Dict[str, Dict[str, float]]:
    g = read_chunks(gold_txt)
    p = read_chunks(pred_txt)

    gb = [c.boundary for c in g]
    pb = [c.boundary for c in p]

    out: Dict[str, Dict[str, float]] = {}
    for mode in ("strict", "lax"):
        strict = mode == "strict"
        tp, fp, fn = greedy_match(gb, pb, strict, th)
        out[f"boundary_{mode}"] = prf(tp, fp, fn)

        scores: List[float] = []
        for gc in g:
            i = best_match_idx(gc.boundary, p, strict, th)
            scores.append(bleu(p[i].text, gc.text) if i is not None else 0.0)
        out[f"avg_chunk_bleu_{mode}"] = {
            "avg_bleu": (sum(scores) / len(scores) if scores else 0.0),
            "n": float(len(scores)),
            "chunks pred/gold": f"{float(len(p))}/{float(len(g))}"
        }
    return out


# =========================
# Chunk alignment eval
# =========================
def load_matches(p: Path) -> Dict[int, List[int]]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[int, List[int]] = {}

    def add(src, tgt):
        s = int(src)
        if isinstance(tgt, list):
            out.setdefault(s, []).extend(int(x) for x in tgt)
        else:
            out.setdefault(s, []).append(int(tgt))

    def handle(src, m):
        if isinstance(m, dict):
            if "ja_idx" in m: return add(src, m["ja_idx"])
            if "tgt_idx" in m: return add(src, m["tgt_idx"])
            if "target" in m: return add(src, m["target"])
        return add(src, m)

    def parse_item(it):
        # [src, tgt]
        if isinstance(it, (list, tuple)) and len(it) == 2:
            return add(it[0], it[1])

        if not isinstance(it, dict):
            raise ValueError(f"Unsupported item: {type(it)} / {it}")

        # flat: {"src":..,"tgt":..} / {"source":..,"target":..} / {"s":..,"t":..}
        for sk, tk in (("src", "tgt"), ("source", "target"), ("s", "t")):
            if sk in it and tk in it:
                return add(it[sk], it[tk])

        # nested: {"en_idx"/"src_idx":.., "matches":[...]}
        if "matches" in it and ("en_idx" in it or "src_idx" in it):
            src = it.get("en_idx", it.get("src_idx"))
            ms = it.get("matches", [])
            if not isinstance(ms, list):
                raise ValueError(f"matches must be list: {type(ms)}")
            for m in ms:
                handle(src, m)
            return

        raise ValueError(f"Unsupported dict item: {it}")

    if isinstance(obj, dict):
        # root dict: {"0":[3], ...}  or single nested dict
        if "matches" in obj and ("en_idx" in obj or "src_idx" in obj):
            parse_item(obj)
        else:
            for k, v in obj.items():
                add(k, v)

    elif isinstance(obj, list):
        for it in obj:
            parse_item(it)

    else:
        raise ValueError(f"Unsupported root type: {type(obj)}")

    # uniq + sort
    for k in list(out.keys()):
        out[k] = sorted(set(out[k]))
    return out

def concat(chunks: Sequence[Chunk], idxs: Sequence[int]) -> str:
    parts = []
    for i in idxs:
        if 0 <= i < len(chunks):
            parts.append(chunks[i].text)
    return "\n\n".join(parts).strip()

def eval_chunk_align(gold_matches: Path, pred_matches: Path,
    gold_src_txt: Path, gold_tgt_txt: Path,
    pred_src_txt: Path, pred_tgt_txt: Path,
    th: float,) -> Dict[str, Dict[str, float]]:
    
    gS, gT = read_chunks(gold_src_txt), read_chunks(gold_tgt_txt)
    pS, pT = read_chunks(pred_src_txt), read_chunks(pred_tgt_txt)
    GM, PM = load_matches(gold_matches), load_matches(pred_matches)

    res: Dict[str, Dict[str, float]] = {}
    for mode in ("strict", "lax"):
        strict = mode == "strict"

        # gold src -> pred src by boundary (prevents gaming by over/under-splitting on src)
        g2p: Dict[int, List[int]] = {}
        for gi in range(len(gS)):
            pi = best_match_idx(gS[gi].boundary, pS, strict, th)
            g2p[gi] = [pi] if pi is not None else []
            
        pred_pairs_mapped = 0

        gold_tgt_bounds: List[str] = []
        pred_tgt_bounds: List[str] = []
        bleu_scores: List[float] = []

        for g_src_i, g_tgt_is in GM.items():
            if not (0 <= g_src_i < len(gS)):
                continue

            # gold target (1:n) and boundaries
            g_text = concat(gT, g_tgt_is)
            gold_tgt_bounds += [gT[i].boundary for i in g_tgt_is if 0 <= i < len(gT)]

            # predicted target indices via mapped pred src -> pred matches
            p_tgt_is: List[int] = []
            for p_src_i in g2p.get(g_src_i, []):
                p_tgt_is += PM.get(p_src_i, [])

            if p_tgt_is:
                pred_pairs_mapped += 1
                pred_tgt_bounds += [pT[i].boundary for i in p_tgt_is if 0 <= i < len(pT)]
                p_text = concat(pT, p_tgt_is)
                bleu_scores.append(bleu(p_text, g_text))
            else:
                bleu_scores.append(0.0)

        tp, fp, fn = greedy_match(gold_tgt_bounds, pred_tgt_bounds, strict, th)
        res[f"target_boundary_{mode}"] = prf(tp, fp, fn)
        res[f"avg_target_chunk_bleu_{mode}"] = {
            "avg_bleu": (sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0),
            "n": float(len(bleu_scores)),
            "aligned chunks pred/gold": f"{float(pred_pairs_mapped)}/{float(len(gold_tgt_bounds))}",
        }
    return res


# =========================
# Sentence alignment eval
# =========================
def read_pairs(en: Path, ja: Path) -> List[Tuple[str, str]]:
    e = en.read_text(encoding="utf-8", errors="replace").splitlines()
    j = ja.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[Tuple[str, str]] = []
    for a, b in zip(e, j):
        a, b = norm(a), norm(b)
        # pairs with empty lines are removed for this evaluation
        if a and b:
            out.append((a, b))
    return out

def eval_sent(g_en: Path, g_ja: Path, p_en: Path, p_ja: Path, th: float) -> Dict[str, Dict[str, float]]:
    G, P = read_pairs(g_en, g_ja), read_pairs(p_en, p_ja)

    # strict set match
    Gs, Ps = set(G), set(P)
    strict = prf(len(Gs & Ps), len(Ps - Gs), len(Gs - Ps))

    # lax greedy match (both src & tgt must pass)
    cand: List[Tuple[int, int, float]] = []
    for gi, (gs, gt) in enumerate(G):
        for pi, (ps, pt) in enumerate(P):
            sc_s = jac(gs, ps)
            sc_t = jac(gt, pt)
            if sc_s >= th and sc_t >= th:
                cand.append((gi, pi, (sc_s + sc_t) / 2))
    cand.sort(key=lambda x: x[2], reverse=True)

    used_g, used_p, tp = set(), set(), 0
    for gi, pi, _ in cand:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1
    lax = prf(tp, len(P) - tp, len(G) - tp)

    return {"strict": strict, "lax": lax, "counts": {"gold_pairs": float(len(G)), "pred_pairs": float(len(P))}}


# =========================
# Run pipeline (direct calls)
# =========================
def run_pipeline(evalset, is_ocr_only, align_method, threshs_height_r):
    """
    Run chunk detect -> (optional) chunk align -> sentence align,
    using your imported run_* functions (NO subprocess).
    """
    setup_logger()

    # mirror your main.py behavior: clear result/
    if RES.exists():
        shutil.rmtree(RES)
    RES.mkdir(parents=True, exist_ok=True)

    pdf_dir = str(DATA_PDF / f"{evalset}")

    if evalset == "unesco" or "shuffled_unesco":
        langs = ["en", "ja"]
        model_ocr = ["tesseract", "yomitoku"]
        path_yolo_src = "tools/yolov12s-doclaynet.pt"
        path_yolo_tgt = "tools/yolov12s-doclaynet-ja.pt"
        reading_direction = ["left-right", "left-right"]
        isvertical = [False, False]
        threshs_height_r = [float(threshs_height_r[0]), float(threshs_height_r[1])]
    else:
        raise ValueError(f"Unknown evalset: {evalset}")

    run_chunk_detection(
        pdf_dir=pdf_dir,
        langs=langs,
        out_dir=".",
        model_ocr=model_ocr,
        model_od="yolo",
        path_ndl_lite="tools/ndlkotenocr-lite",
        path_ndlocr="tools/ndlocr_cli",
        path_yolo=[path_yolo_src, path_yolo_tgt],
        dpi=300,
        threshs_height_r=threshs_height_r,
        reading_direction=reading_direction,
        isvertical=isvertical,
        is_ocr_only=is_ocr_only,
    )

    run_chunk_alignment(
        langs=langs,
        mode="align",
        thresh_chunk_sim=0.3,
        model_emb="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        path_reg_weights="tools/chunkalign_reg_weights.json",
    )

    run_sentence_alignment(
        src_lang=langs[0],
        tgt_lang=langs[1],
        align_method=align_method,  # "vecalign" / "bleualign"
        model_emb="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_nllb="facebook/nllb-200-distilled-600M",
        skip_mt=False,
    )


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", choices=["unesco", "shuffled_unesco"], required=True)
    ap.add_argument("--align_method", choices=["vecalign", "bleualign"], default="vecalign")
    ap.add_argument("--threshs_height_r", nargs=2, type=float, default=[1.4, 1.4])
    ap.add_argument("--is_ocr_only", action="store_true")
    ap.add_argument("--jaccard_thresh", type=float, default=0.9)
    ap.add_argument("--skip_run", action="store_true")
    args = ap.parse_args()

    # run pipeline
    if not args.skip_run:
        run_pipeline(args.evalset, args.is_ocr_only, args.align_method, args.threshs_height_r)

    print("\nEvaluating ...\n")
    # chunk detection eval
    gold_en = GOLD / f"{args.evalset}_all_chunks_ordered-en.txt"
    gold_ja = GOLD / f"{args.evalset}_all_chunks_ordered-ja.txt"
    pred_en = RES / "chunks-en/all_chunks_ordered.txt"
    pred_ja = RES / "chunks-ja/all_chunks_ordered.txt"

    report: Dict[str, object] = {
        "meta": {
            "evalset": args.evalset,
            "align_method": args.align_method,
            "is_ocr_only": bool(args.is_ocr_only),
            "jaccard_thresh": float(args.jaccard_thresh),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        },
        "chunk_detection": {
            "en": eval_chunk_det(gold_en, pred_en, args.jaccard_thresh),
            "ja": eval_chunk_det(gold_ja, pred_ja, args.jaccard_thresh),
        },
        "chunk_alignment": None,
        "sentence_alignment": None,
    }

    # chunk alignment eval
    if not args.is_ocr_only:
        report["chunk_alignment"] = eval_chunk_align(
            GOLD / f"{args.evalset}_matches.json",
            RES / "chunks_aligned/matches.json",
            gold_en,
            gold_ja,
            pred_en,
            pred_ja,
            args.jaccard_thresh,
        )

    # sentence alignment eval
    report["sentence_alignment"] = eval_sent(
        GOLD / f"{args.evalset}_aligned-en.txt",
        GOLD / f"{args.evalset}_aligned-ja.txt",
        RES / "sentences_aligned/aligned-en.txt",
        RES / "sentences_aligned/aligned-ja.txt",
        args.jaccard_thresh,
    )

    # print(json.dumps(report, ensure_ascii=False, indent=2))
    print_eval_summary(report)
    outdir = RES / "eval"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.evalset}_{args.align_method}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Saved]", out)


if __name__ == "__main__":
    main()