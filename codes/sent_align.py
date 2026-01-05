from __future__ import print_function
import shutil, sys, re, time, threading, subprocess
from pathlib import Path
import torch
from fugashi import Tagger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
sys.path.append("./tools/Bleualign")
from bleualign.align import Aligner
from codes.util import Embedder
import pickle
import numpy as np


##############
##bluealign##
##############

# util
def nllb_codes():
    return {
        "ar": "arb_Arab",    # Arabic
        "de": "deu_Latn",    # German
        "en": "eng_Latn",    # English
        "es": "spa_Latn",    # Spanish
        "fr": "fra_Latn",    # French
        "hi": "hin_Deva",    # Hindi
        "it": "ita_Latn",    # Italian
        "ja": "jpn_Jpan",    # Japanese
        "ko": "kor_Hang",    # Korean
        "pt": "por_Latn",    # Portuguese
        "ru": "rus_Cyrl",    # Russian
        "zh": "zho_Hans",    # Simplified Chinese
        "zh-trad": "zho_Hant",  # Traditional Chinese
        
        # ADD YOUR LANGUAGES
    }

_JA = Tagger()
def ja_tok_line(s):
    s = (s or "").strip()
    if not s:
        return ""
    return " ".join(w.surface for w in _JA(s))

def ja_detok_line(s):
    return (s or "").replace(" ", "")

def read_lines(p):
    with Path(p).open("r", encoding="utf-8", newline="") as f:
        return f.readlines()

def write_lines(p, lines):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        for s in lines:
            if s is None:
                s = ""
            if s.endswith("\n"):
                f.write(s)
            else:
                f.write(s + "\n")

# machine translation
class NLLBTranslator:
    """
    NLLB (Fast) 
    """
    def __init__(self, model_name, src_lang, tgt_lang):
        code = nllb_codes()
        self.src = code.get(src_lang, src_lang)
        self.tgt = code.get(tgt_lang, tgt_lang)
        # give src/tgt_lang
        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=self.src,
            tgt_lang=self.tgt,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        try:
            bos_id = self.tok.convert_tokens_to_ids(self.tgt)
            if isinstance(bos_id, int) and bos_id >= 0:
                self.model.config.forced_bos_token_id = bos_id
        except Exception:
            pass

    def translate(self, src_dir, out_dir, batch_size=8, max_new_tokens=128):
        src_dir = Path(src_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in sorted(src_dir.glob("*.txt")):
            lines = read_lines(f)
            outs = []
            for i in range(0, len(lines), batch_size):
                chunk = lines[i:i+batch_size]

                self.tok.set_src_lang_special_tokens(self.src)
                enc = self.tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.model.device)

                self.tok.set_tgt_lang_special_tokens(self.tgt)

                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens
                )
                outs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))

            out_path = out_dir / f.name
            write_lines(out_path, outs)

# preprocess for Bleualign
def prepare_for_bleualign(src_file, tgt_file, src_trans_file, src_lang, tgt_lang):
    src_file = Path(src_file)
    tgt_file = Path(tgt_file)
    src_trans_file = Path(src_trans_file)
    src_tok = src_file.with_name(f"{src_file.stem}-tok{src_file.suffix}")
    src_trans_tok = src_trans_file.with_name(f"{src_trans_file.stem}-tok{src_trans_file.suffix}")
    tgt_tok = tgt_file.with_name(f"{tgt_file.stem}-tok{tgt_file.suffix}")

    if not src_trans_file.exists():
        raise FileNotFoundError(f"...missing MT: {src_trans_file}")

    if tgt_lang == "ja":
        if not tgt_tok.exists():
            write_lines(tgt_tok, [ja_tok_line(x) for x in read_lines(tgt_file)])
        if not src_trans_tok.exists():
            write_lines(src_trans_tok, [ja_tok_line(x) for x in read_lines(src_trans_file)])
    else:
        if not tgt_tok.exists():
            shutil.copyfile(tgt_file, tgt_tok)
        if not src_trans_tok.exists():
            shutil.copyfile(src_trans_file, src_trans_tok)
    
    if src_lang == "ja":
        if not src_tok.exists():
            write_lines(src_tok, [ja_tok_line(x) for x in read_lines(src_file)])
    else:
        if not src_tok.exists():
            shutil.copyfile(src_file, src_tok)

    return src_tok, src_trans_tok, tgt_tok

def bleualign_one(src_file, tgt_file, srctotarget_file, src_aligned, tgt_aligned):
    options = {
        'srcfile': str(src_file),
        'targetfile': str(tgt_file),
        'srctotarget': [str(srctotarget_file)],
        'targettosrc': [],
        'output-src': src_aligned, 
        'output-target': tgt_aligned,
        }
    a = Aligner(options)
    a.mainloop()

# process
def run_bleualign(src_dir, tgt_dir, out_dir, src_lang, tgt_lang, mt, model_nllb, skip_mt=False):
    trans_dir = src_dir / "mt-translation"

    # util(spinner)
    def spinner_task(msg: str):
        spinner = "|/-\\"
        i = 0
        while not done:
            sys.stdout.write(f"\r{msg} {spinner[i % len(spinner)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    # create translations by MT
    if not skip_mt:
        done = False
        t = threading.Thread(target=spinner_task, args=("Translating source texts by MT(NLLB) for Bleualign ....",))
        t.start()
        # print("Activating MT(NLLB) for Bleualign ....")
        tr = NLLBTranslator(model_nllb, src_lang, tgt_lang) if mt == "nllb" else None
        if tr:
            # store in trans_dir
            tr.translate(src_dir, trans_dir, batch_size=8, max_new_tokens=128)
        done = True
        t.join()
        print("\nMT done ....")
    else:
        any_src = next(src_dir.glob("*.txt"), None)
        if any_src and not any_src.with_suffix(".trans").exists():
            raise SystemExit(f'If --skip-mt True,  please set -trans.txt file in {src_dir}')

    # align per chunk
    for src_file in sorted(src_dir.glob("*.txt")):
        name = src_file.name
        src_trans_file = trans_dir / name
        tgt_file = tgt_dir / name
        
        if not tgt_file.exists():
            print("...Coudn't find a target file")
            continue
        
        # tokenize agglutinating/isolating languages (japanese only so far)
        src_tok, src_trans_tok, tgt_tok = prepare_for_bleualign(src_file, tgt_file, src_trans_file, src_lang, tgt_lang)
        use_s = src_tok if src_lang == "ja" else src_file 
        use_tr = src_trans_tok if tgt_lang == "ja" else src_trans_file
        use_t = tgt_tok if tgt_lang == "ja" else tgt_file
        print("Tokenization done ....\n")
        
        # bleualign
        src_aligned = out_dir / f"{src_file.stem}-aligned-{src_lang}.txt"
        tgt_aligned = out_dir / f"{tgt_file.stem}-aligned-{tgt_lang}.txt"
        print("=====Bleualign======")
        bleualign_one(use_s, use_t, use_tr, src_aligned, tgt_aligned)

    # concat
    src_all = out_dir / f"aligned-{src_lang}.txt"
    tgt_all = out_dir / f"aligned-{tgt_lang}.txt"
    # source
    SRC = []
    for s_file in sorted(
            out_dir.glob(f"*-aligned-{src_lang}.txt"),
            key=lambda p: int(re.search(r"chunk(\d+)-aligned-", p.name).group(1))):
        SRC.extend(read_lines(s_file))
    write_lines(src_all, SRC)
    TGT = []
    # for t_file in sorted(out_dir.glob(f"*-aligned-{tgt_lang}.txt")):
    for t_file in sorted(
            out_dir.glob(f"*-aligned-{tgt_lang}.txt"),
            key=lambda p: int(re.search(r"chunk(\d+)-aligned-", p.name).group(1))):
        lines = read_lines(t_file)
        if tgt_lang == "ja":
            lines = [ja_detok_line(t) if t else t for t in lines]
        TGT.extend(lines)
    write_lines(tgt_all, TGT)

    print(f"[sent-align/vecalign] pairs={len(SRC)} -> {src_all.name}, {tgt_all.name}")



##############
##vecalign##
##############

def run_vecalign(src_dir, tgt_dir, out_dir, src_lang, tgt_lang, model_emb):
    vecalign_dir = out_dir / "vecalign_result"
    vecalign_dir.mkdir(parents=True, exist_ok=True)

    s_o = vecalign_dir / "src_overlap.txt"
    t_o = vecalign_dir / "tgt_overlap.txt"

    # ----------------------------
    # 1) overlap（全チャンクまとめて1個）
    # ----------------------------
    src_files = [str(p) for p in sorted(src_dir.glob("*.txt"))]
    tgt_files = [str(p) for p in sorted(tgt_dir.glob("*.txt"))]
    
    subprocess.run(
        [sys.executable, "tools/vecalign/overlap.py", "-i", *src_files, "-o", str(s_o), "-n", "10"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "tools/vecalign/overlap.py", "-i", *tgt_files, "-o", str(t_o), "-n", "10"],
        check=True,
    )

    # overlap を読み込む（1行=1フレーズ）
    src_overlap_txt = read_lines(s_o)  # ["行1", "行2", ...]
    tgt_overlap_txt = read_lines(t_o)

    # ----------------------------
    # 2) create embeddings
    # ----------------------------
    embedder = Embedder(model_emb)
    E_src = embedder.encode(src_overlap_txt)
    E_tgt = embedder.encode(tgt_overlap_txt)

    # float32 binary にする
    E_src = np.asarray(E_src, dtype=np.float32)
    E_tgt = np.asarray(E_tgt, dtype=np.float32)

    s_o_emb = vecalign_dir / "src_overlap_emb.bin"
    t_o_emb = vecalign_dir / "tgt_overlap_emb.bin"

    # binary write（vecalign の仕様どおり）
    E_src.tofile(s_o_emb)
    E_tgt.tofile(t_o_emb)

    # ----------------------------
    # 3) align per chunk-pair
    # ----------------------------
    for src_file in sorted(src_dir.glob("*.txt")):
        name = src_file.name
        tgt_file = tgt_dir / name

        if not tgt_file.exists():
            print(f"[vecalign] Couldn't find a target file: {tgt_file}")
            continue

        out_pkl = vecalign_dir / f"{src_file.stem}.pkl"

        subprocess.run(
            [
                sys.executable,
                "tools/vecalign/vecalign.py",
                "--alignment_max_size", "8",
                "--src", str(src_file),
                "--tgt", str(tgt_file),
                "--src_embed", str(s_o), str(s_o_emb),
                "--tgt_embed", str(t_o), str(t_o_emb),
                "--debug_save_stack", str(out_pkl),
            ],
            check=True,
        )

        # ----------------------------
        # 4) convert pickle -> aligned txt
        # ----------------------------
        sys.path.insert(0, "tools/vecalign")
        with open(out_pkl, "rb") as f:
            data = pickle.load(f)

        aligned_pairs = data[0][0]["final_alignments"]

        src_lines = read_lines(src_file)
        tgt_lines = read_lines(tgt_file)

        src_aligned = out_dir / f"{src_file.stem}-aligned-{src_lang}.txt"
        tgt_aligned = out_dir / f"{tgt_file.stem}-aligned-{tgt_lang}.txt"

        out_src_lines = []
        out_tgt_lines = []

        for src_ids, tgt_ids in aligned_pairs:
            s = " ".join(src_lines[i].rstrip("\n") for i in src_ids) if len(src_ids) > 0 else ""
            t = " ".join(tgt_lines[i].rstrip("\n") for i in tgt_ids) if len(tgt_ids) > 0 else ""
            out_src_lines.append(s)
            out_tgt_lines.append(t)

        write_lines(src_aligned, out_src_lines)
        if tgt_lang == "ja":
            out_tgt_lines = [ja_detok_line(t) if t else t for t in out_tgt_lines]
        write_lines(tgt_aligned, out_tgt_lines)


    # ----------------------------
    # 5) concat（全チャンクの aligned txt をくっつける）
    # ----------------------------
    src_all = out_dir / f"aligned-{src_lang}.txt"
    tgt_all = out_dir / f"aligned-{tgt_lang}.txt"

    SRC = []
    # for s_file in sorted(out_dir.glob(f"*-aligned-{src_lang}.txt")):
    for s_file in sorted(
            out_dir.glob(f"*-aligned-{src_lang}.txt"),
            key=lambda p: int(re.search(r"chunk(\d+)-aligned-", p.name).group(1))):
        SRC.extend(read_lines(s_file))
    write_lines(src_all, SRC)
    TGT = []
    # for t_file in sorted(out_dir.glob(f"*-aligned-{tgt_lang}.txt")):
    for t_file in sorted(
            out_dir.glob(f"*-aligned-{tgt_lang}.txt"),
            key=lambda p: int(re.search(r"chunk(\d+)-aligned-", p.name).group(1))):
        lines = read_lines(t_file)
        if tgt_lang == "ja":
            lines = [ja_detok_line(t) if t else t for t in lines]
        TGT.extend(lines)
    write_lines(tgt_all, TGT)

    print(f"[sent-align/vecalign] pairs={len(SRC)} -> {src_all.name}, {tgt_all.name}")