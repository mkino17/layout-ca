from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
from codes.chunk_align import run_chunkalign, run_train


def run_chunk_alignment(langs, mode, path_reg_weights, model_emb, thresh_chunk_sim):
    if mode == "align":
        src_chunks_dir=Path(f"result/chunks-{langs[0]}")
        tgt_chunks_dir=Path(f"result/chunks-{langs[1]}")
        out_json = Path("result/chunks_aligned/matches.json")
        out_txt_dir = Path("result/chunks_aligned")
        # chunkalign
        run_chunkalign(src_chunks_dir, tgt_chunks_dir, path_reg_weights, model_emb, 
                       thresh_chunk_sim, out_json, out_txt_dir, langs)
        return str(out_json)

    run_train(
        model_emb=model_emb,
        out_path=path_reg_weights,
        langs=langs
    )
    return str(path_reg_weights)