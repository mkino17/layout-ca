from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from codes.data_loader import pdf_to_jpg_folder
from codes.ocr_object_detection import run_ocr_object_detection
from codes.reading_order_detection import run_reading_order_detection
from codes.chunk_split import process_chunk_split


def _run_for_lang(out_dir, lang, model_ocr, model_od, path_yolo, 
                    threshs_height_r, reading_direction, isvertical, is_ocr_only, ocr_device="cuda"):

    jpg_dir      = Path(out_dir) / "jpg_pages" / lang
    tokens_dir   = Path(out_dir) / "result" / f"tokens-{lang}"
    segments_dir = Path(out_dir) / "result" / f"segments-{lang}"
    ordered_dir  = Path(out_dir) / "result" / f"segments_ordered-{lang}"
    chunks_dir   = Path(out_dir) / "result" / f"chunks-{lang}"
    
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # OCR + Object Detection
    if jpg_dir.exists():
        run_ocr_object_detection(jpg_dir, lang, tokens_dir, segments_dir, model_ocr, model_od,
                                path_yolo, is_ocr_only, ocr_device)
    else:
        print(f"[WARN] {jpg_dir} not found, skipping OCR for {lang}.")
    
    # Reading Order
    if segments_dir.exists():
        all_pages_ordered = run_reading_order_detection(segments_dir, ordered_dir, reading_direction, isvertical)
        # Chunk Split
        out_json, out_json_ordered = process_chunk_split(
            all_pages_ordered,
            chunks_dir / "all_chunks.json",
            chunks_dir / "all_chunks_ordered.json",
            chunks_dir / "all_chunks_ordered.txt",
            chunks_dir / "all_chunks_ordered_label.txt",
            threshs_height_r
        )
        return {"aggregated": out_json, "ordered": out_json_ordered}
    else:
        print(f"[WARN] {segments_dir} not found, skipping reading-order/chunk-split for {lang}.")
        return None

# main
def run_chunk_detection(pdf_dir, out_dir, model_ocr, model_od, 
                        path_yolo, dpi, langs, threshs_height_r, reading_direction, isvertical, is_ocr_only):

    # PDF → JPG
    pdf_to_jpg_folder(pdf_dir, Path(out_dir)/"jpg_pages", dpi, langs)
    # Each lang
    outs = {}
    for i, lang in enumerate(langs):
        outs[lang] = _run_for_lang(out_dir, lang, model_ocr[i], model_od, path_yolo[i], 
                                   threshs_height_r[i], reading_direction[i], isvertical[i], is_ocr_only)
    return outs