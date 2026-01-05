import argparse, shutil
from pathlib import Path
from codes.util import setup_logger, save_config, load_config
from model.chunk_detection import run_chunk_detection
from model.chunk_alignment import run_chunk_alignment
from model.sentence_alignment import run_sentence_alignment
CONFIG_PATH = Path("./tools/config.json")


def build_parser():
    cfg = load_config(CONFIG_PATH)
    p = argparse.ArgumentParser("alignflow-folder")
    sub = p.add_subparsers(dest="cmd")

    # Chunk Detection
    p1 = sub.add_parser("chunk-detect")
    p1.add_argument("--pdf_dir", default="data")
    p1.add_argument("--langs", nargs=2, default=cfg.get("langs", ["en","ja"]))
    p1.add_argument("--model_ocr", nargs=2, default=["tesseract", "yomitoku"], choices=["tesseract", "yomitoku"])
    p1.add_argument("--model_od", default="yolo", choices=["yolo"])
    p1.add_argument("--dpi", type=int, default=300)
    p1.add_argument("--path_yolo_src", default="tools/yolov12s-doclaynet.pt")
    p1.add_argument("--path_yolo_tgt", default="tools/yolov12s-doclaynet-ja.pt")
    p1.add_argument("--threshs_height_r", nargs=2, default=[1.3, 1.3], 
                    help="threshold ratio of (height of section header)/(height of text) for chunk split: [src, tgt]")
    p1.add_argument("--reading_direction", nargs=2, default=["left-right","left-right"])
    p1.add_argument("--isvertical", nargs=2, default=[False, False])
    p1.add_argument("--is_ocr_only", default=False)
    
    # Chunk Alignment
    p2 = sub.add_parser("chunk-align")
    p2.add_argument("--mode", choices=["align", "train"], default="align")
    p2.add_argument("--langs", nargs=2, default=cfg.get("langs", ["en","ja"]))
    p2.add_argument("--model_emb", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    p2.add_argument("--path_reg_weights", default="tools/chunkalign_reg_weights.json")
    p2.add_argument("--thresh_chunk_sim", type=float, default=0.3)

    # Sentence Alignment
    p3 = sub.add_parser("sent-align")
    p3.add_argument("--langs", nargs=2, default=cfg.get("langs", ["en","ja"]))
    p3.add_argument("--mt", nargs=2, choices=["nllb", "none"], default="nllb")
    p3.add_argument("--align_method", choices=["vecalign", "bleualign"], default="vecalign")
    p3.add_argument("--model_emb", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    p3.add_argument("--model_nllb", default="facebook/nllb-200-distilled-600M")
    p3.add_argument("--skip_mt", action="store_true")

    return p


def main():
    setup_logger()
    args = build_parser().parse_args()

    if args.cmd == "chunk-detect":
        
        result_dir = Path("result")
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nRunning chunk detection ....\n")
        save_config(CONFIG_PATH, langs=args.langs)
        run_chunk_detection(
            pdf_dir=args.pdf_dir, 
            langs=args.langs,
            out_dir=".", 
            model_ocr = args.model_ocr,
            model_od = args.model_od,
            path_yolo=[args.path_yolo_src, args.path_yolo_tgt],
            dpi=args.dpi,
            threshs_height_r=args.threshs_height_r,
            reading_direction=args.reading_direction,
            isvertical=args.isvertical,
            is_ocr_only=args.is_ocr_only
            )
        print("\nComplete\nSaved: result/tokens, segments, chunks\n")

    elif args.cmd == "chunk-align":
        
        print("\nRunning chunk alignment ....\n")
        run_chunk_alignment(
            langs = args.langs,
            mode=args.mode,
            thresh_chunk_sim = args.thresh_chunk_sim,
            model_emb=args.model_emb,
            path_reg_weights=args.path_reg_weights,
            )
        if args.mode == "align":
            print("\nComplete\nSaved: result/chunks_aligned\n")
        else:
            print(f"\nComplete\nSaved: {args.path_reg_weights}\n")

    elif args.cmd == "sent-align":
        
        print("\nRunning sentence alignment ....\n")
        run_sentence_alignment(
            src_lang=args.langs[0],
            tgt_lang=args.langs[1],
            align_method=args.align_method,
            model_emb=args.model_emb,
            model_nllb=args.model_nllb,
            skip_mt=args.skip_mt
            )
        print("\nComplete\nSaved: result/sentences_aligned\n")

    else:
        print("Use: chunk-detect | chunk-align | sent-align | all")


if __name__ == "__main__":
    main()