import json
import re
from pathlib import Path
from typing import List, Dict, Any

HEAD_LABELS = {"Title", "Section-header", "Header"}
ALL_LABELS  = ["Title","Section-header","Text","List-item","Formula",
               "Picture","Table","Caption","Footnote","Page-header","Page-footer"]
CHUNK_MARK = "<<<CHUNK>>>"

def should_start_chunk(seg, thresh_height_r):
    return (seg.get("label") in HEAD_LABELS) and (seg.get("hratio_to_text") is not None) and (seg["hratio_to_text"] >= thresh_height_r)

def split_into_chunks(all_segs, thresh_height_r):
    chunks, cur = [], []
    for s in all_segs:
        if should_start_chunk(s, thresh_height_r):
            if cur: chunks.append(cur)
            cur = [s]
        else:
            cur.append(s)
    if cur: chunks.append(cur)
    return chunks

def aggregate_labels_all(chunk_segments):
    labels = {lbl: {"ids": [], "text": "", "count": 0} for lbl in ALL_LABELS}
    for s in chunk_segments:
        key = s.get("label") or ""
        labels.setdefault(key, {"ids": [], "text": "", "count": 0})
        labels[key]["ids"].append(s.get("id"))
        t = (s.get("text") or "").strip()
        if t:
            labels[key]["text"] = ((labels[key]["text"] + " " + t).strip()) if labels[key]["text"] else t
        labels[key]["count"] += 1
    return labels

def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def chunk_text(segs: List[Dict[str, Any]]) -> str:
    texts = [norm(s.get("text")) for s in segs]
    texts = [t for t in texts if t]
    return "\n".join(texts).strip()

def chunk_text_labeled(segs: List[Dict[str, Any]]) -> str:
    lines = []
    for s in segs:
        lab = s.get("label") or "UNK"
        txt = norm(s.get("text"))
        if not txt: continue
        lines.append(f"<{lab}> {txt}")
    return "\n".join(lines).strip()

# main
def process_chunk_split(in_file: Path, out_json: Path, out_json_ordered: Path, out_txt: Path, out_txt_label: Path, thresh_height_r):
    all_segs = json.loads(in_file.read_text(encoding="utf-8"))
    chunks_segs = split_into_chunks(all_segs, thresh_height_r)

    # .json
    out_chunks = [{"chunk_index": i, "labels": aggregate_labels_all(segs)} for i, segs in enumerate(chunks_segs)]
    out_json.write_text(json.dumps({"n_chunks": len(out_chunks), "thresh_height_r": thresh_height_r, "chunks": out_chunks}, 
                                   ensure_ascii=False, indent=2), encoding="utf-8")

    chunks_in_order = [{"chunk_index": i, "segments": segs}for i, segs in enumerate(chunks_segs)]
    out_json_ordered.write_text(json.dumps({"n_chunks": len(chunks_in_order), "thresh_height_r": thresh_height_r, "chunks": chunks_in_order}, 
                                           ensure_ascii=False, indent=2), encoding="utf-8")
    
    # .txt
    chunks_texts = [chunk_text(segs) for segs in chunks_segs]
    texts = f"\n{CHUNK_MARK}\n".join(chunks_texts)
    out_txt.write_text(texts, encoding="utf-8")

    # .txt (labeled)
    chunks_texts_label = [chunk_text_labeled(segs) for segs in chunks_segs]
    texts_label = f"\n{CHUNK_MARK}\n".join(chunks_texts_label)
    out_txt_label.write_text(texts_label, encoding="utf-8")

    return out_json, out_json_ordered