import cv2, json
from pathlib import Path
from statistics import mean
import pytesseract
from PIL import Image
import logging
logging.getLogger("yomitoku.base").setLevel(logging.WARNING)
from yomitoku import OCR
from ultralytics import YOLO

tesseract_codes = {
    "ar": "ara",       # Arabic
    "de": "deu",       # German
    "en": "eng",       # English
    "es": "spa",       # Spanish
    "fr": "fra",       # French
    "hi": "hin",       # Hindi
    "it": "ita",       # Italian
    "ja": "jpn",       # Japanese
    "ko": "kor",       # Korean
    "pt": "por",       # Portuguese
    "ru": "rus",       # Russian
    "zh": "chi_sim",   # Simplified Chinese
    "zh-trad": "chi_tra", # Traditional Chinese
    # ADD YOUR LANGUAGES
    }


# OCR: Tesseract
def ocr_tesseract(img_bgr, ocr_lang):
    image_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ocr_data = pytesseract.image_to_data(image_pil, lang=ocr_lang, output_type=pytesseract.Output.DICT)
    toks = []
    for i in range(len(ocr_data["text"])):
        content = ocr_data["text"][i].strip()
        if content == "":
            continue
        left = int(ocr_data["left"][i]) ; top = int(ocr_data["top"][i])
        width = int(ocr_data["width"][i]) ; height = int(ocr_data["height"][i])
        bbox = [left, top, left + width, top + height]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        conf = int(ocr_data["conf"][i])
        r = height / width
        toks.append({
            "text": content, "bbox": bbox, "conf": conf,
            "h": min(width, height), "_center": center, "r": r
            })
    return toks

# OCR: Yomitoku
def ocr_yomitoku(img_bgr, ocr_obj):
    ocr_data = ocr_obj(img_bgr)[0]
    toks = []
    if not getattr(ocr_data, "words", None):
        return toks
    for w in ocr_data.words:
        content = (getattr(w, "content", "") or "").strip()
        if not content: 
            continue
        xs = [p[0] for p in w.points]; ys = [p[1] for p in w.points]
        x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        center = [(bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0]
        det = float(getattr(w, "det_score", 1.0) or 1.0)
        rec = float(getattr(w, "rec_score", 1.0) or 1.0)
        conf = max(0.0, min(1.0, det * rec))
        r = (bbox[3]-bbox[1]) / (bbox[2]-bbox[0])
        toks.append({
            "text": content, "bbox": bbox, "conf": conf,
            "h": min(bbox[2]-bbox[0], bbox[3]-bbox[1]), 
            "_center": center, "r": r
        })
    return toks

# util
def _split_horiz_vert(toks_in):
    horiz, vert = [], []
    avg_r = mean([t["r"] for t in toks_in])
    for t in toks_in:
        # x1, y1, x2, y2 = t["bbox"]
        # # w, h = (x2 - x1), (y2 - y1)
        if avg_r >= 1.1:
            vert.append(t)
        else:
            horiz.append(t)
    return horiz, vert

def _sort_horiz(toks):
    if not toks:
        return []
    avg_h = mean([t["h"] for t in toks])
    line_tol = avg_h * 0.5

    toks = sorted(toks, key=lambda t: (t["_center"][1], t["_center"][0]))
    out, cur, last_y = [], [], None
    for t in toks:
        if last_y is None or abs(t["_center"][1] - last_y) <= line_tol:
            cur.append(t)
        else:
            out.extend(sorted(cur, key=lambda tt: tt["_center"][0]))
            cur = [t]
        last_y = t["_center"][1]
    if cur:
        out.extend(sorted(cur, key=lambda tt: tt["_center"][0]))
    return out

def _sort_vert(toks):
    if not toks:
        return []
    widths = [(t["bbox"][2] - t["bbox"][0]) for t in toks]
    avg_w = mean(widths) if widths else 10
    col_tol = max(4, 0.8 * avg_w)

    toks_sorted = sorted(toks, key=lambda t: t["_center"][0])
    columns, cur, last_x = [], [], None
    for t in toks_sorted:
        cx = t["_center"][0]
        if last_x is None or abs(cx - last_x) <= col_tol:
            cur.append(t)
        else:
            columns.append(cur)
            cur = [t]
        last_x = cx
    if cur:
        columns.append(cur)

    columns = [sorted(col, key=lambda t: t["_center"][1]) for col in columns]
    columns = sorted(columns, key=lambda col: mean([t["_center"][0] for t in col]), reverse=True)

    out = []
    for col in columns:
        out.extend(col)
    return out

# process (one page)
def process_page(img_path, ocr, od, tokens_dir, segments_dir, lang, path_yolo, is_ocr_only, text_h_ratio=120):
    img_path = Path(img_path)
    tokens_dir = Path(tokens_dir)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = Path(segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] cannot read image: {img_path}")
        return

    # --- OCR ---
    model_ocr, ocr_obj = ocr
    
    if model_ocr == "tesseract":
        tokens = ocr_tesseract(img, tesseract_codes[lang])
    elif model_ocr == "yomitoku":
        tokens = ocr_yomitoku(img, ocr_obj)
    else:
        raise ValueError(f"Unknown ocr type: {model_ocr}")
    
    # --- OD ---
    res = od(img)[0]
    dets = []
    is_obb = (getattr(res, "obb", None) is not None and len(res.obb) > 0) \
        or ("obb" in Path(path_yolo).stem.lower())
    if is_obb:
        for i, box in enumerate(res.obb.xyxyxyxy):
            xs = box[:, 0]
            ys = box[:, 1]
            x1, y1 = xs.min().item(), ys.min().item()
            x2, y2 = xs.max().item(), ys.max().item()
            cls_id = int(res.obb.cls[i])
            conf   = float(res.obb.conf[i])
            dets.append((x1, y1, x2, y2, cls_id, conf))
    else:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            dets.append((x1, y1, x2, y2, cls_id, conf))
    
    # if using only ocr, process a page as one text box
    if is_ocr_only:
        page_min_x = min((t["bbox"][0] for t in tokens), default=0)
        page_min_y = min((t["bbox"][1] for t in tokens), default=0)
        page_max_x = max((t["bbox"][2] for t in tokens), default=0)
        page_max_y = max((t["bbox"][3] for t in tokens), default=0)
        dets = [(page_min_x, page_min_y, page_max_x, page_max_y, 0, 1)]
        
    segments, text_token_heights = [], []

    for sid, (x1, y1, x2, y2, cls_id, conf) in enumerate(dets):
        
        label = "Text" if is_ocr_only else od.names[cls_id]
        
        # capture tokens inside the box
        toks_in = [t for t in tokens if x1 <= t["_center"][0] <= x2 and y1 <= t["_center"][1] <= y2]

        # sort
        if toks_in:
            horiz, vert = _split_horiz_vert(toks_in)
            ordered_h = _sort_horiz(horiz)
            ordered_v = _sort_vert(vert)
            ordered = ordered_h + ordered_v
            avg_h = mean([t["h"] for t in ordered]) if ordered else None
        else:
            ordered = []
            avg_h = None

        # merge
        if lang == "ja":
            segments.append({
                "id": sid, "label": label, "bbox": [x1,y1,x2,y2], "confidence": round(conf,3),
                "text": "".join([t["text"] for t in ordered]),
                "avg_token_h": (round(avg_h,3) if avg_h is not None else None),
            })
        else:
            segments.append({
                "id": sid, "label": label, "bbox": [x1,y1,x2,y2], "confidence": round(conf,3),
                "text": " ".join([t["text"] for t in ordered]),
                "avg_token_h": (round(avg_h,3) if avg_h is not None else None),
            })

        if label == "Text" and toks_in:
            text_token_heights.extend([t["h"] for t in toks_in])

    # height ratio to average "text"-labelled tokens
    expected_text_h = int(img.shape[0] / text_h_ratio)
    page_text_mean_h = mean(text_token_heights) if text_token_heights else expected_text_h
    for s in segments:
        if page_text_mean_h and s["avg_token_h"] is not None:
            s["hratio_to_text"] = round(s["avg_token_h"]/page_text_mean_h, 4)
        else:
            s["hratio_to_text"] = None

    token_file   = tokens_dir   / f"{img_path.stem}.json"
    segment_file = segments_dir / f"{img_path.stem}.json"

    with open(token_file, "w", encoding="utf-8") as f:
        json.dump([{"text": t["text"], "bbox": t["bbox"], "conf": t["conf"]} for t in tokens], f, ensure_ascii=False, indent=2)
    with open(segment_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Saved: {img_path.name} → tokens:{token_file.name}, segments:{segment_file.name}")


# process (main)
def run_ocr_object_detection(img_dir, lang, tokens_dir, segments_dir, model_ocr, model_od, 
                            path_yolo, is_ocr_only, ocr_device):
    img_dir = Path(img_dir)
    
    # --- OD ---
    if model_od == "yolo":
        od = YOLO(path_yolo)
    else:
        raise ValueError(f"Unknown od_model: {model_od}")
    
    # --- OCR ---
    if model_ocr == "tesseract":
        ocr = (model_ocr, None)
        for img_path in sorted(img_dir.glob("*.jpg")):
            process_page(img_path, ocr, od, tokens_dir, segments_dir, lang, path_yolo, is_ocr_only)

    elif model_ocr == "yomitoku":
        ocr = (model_ocr, OCR(visualize=True, device=ocr_device))
        for img_path in sorted(img_dir.glob("*.jpg")):
            process_page(img_path, ocr, od, tokens_dir, segments_dir, lang, path_yolo, is_ocr_only)
        
    else:
        raise ValueError(f"Unknown model_ocr: {model_ocr}")