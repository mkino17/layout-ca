# coding: utf-8
import re, shutil
from pathlib import Path
from typing import List, Tuple
from sentence_splitter import split_text_into_sentences
from codes.sent_align import run_bleualign
from codes.sent_align import run_vecalign

LABEL_ORDER = [
    "Title","Section-header","Caption","Text","List-item","Formula",
    "Table","Picture","Footnote","Page-header","Page-footer"
    ]
# manual rule for japanese sent split
JA_CLOSERS = '」』】》〉〕］”）)'
JA_BOUNDARY = re.compile(fr'[。！？][{re.escape(JA_CLOSERS)}]*')
JA_TERMINAL = re.compile(fr'[。！？][{re.escape(JA_CLOSERS)}]*\s*$')


# util
def sent_split(text: str, lang: str) -> List[str]:
    text = text.replace("\u3000", " ").strip()
    if not text:
        return []
    # sentence_splitter doesn't cover ja
    if lang == "ja":
        parts, last = [], 0
        for m in JA_BOUNDARY.finditer(text):
            end = m.end(); seg = text[last:end].strip()
            if seg: parts.append(seg); last = end
        tail = text[last:].strip()
        if tail: parts.append(tail)
    else:
        # sentence-splitter
        parts = split_text_into_sentences(text=text, language=lang)
    return [p.strip() for p in parts if p and p.strip()]

def ends_with_terminal(s: str, lang: str) -> bool:
    if lang == "ja":
        return bool(JA_TERMINAL.search(s))
    else:
        return bool(re.search(r"[.!?。！？][\"'”’)\]]*\s*$", s))

def is_numeric_or_roman_only(s: str) -> bool:
    """
    A text full of numbers are exceptions for concat as it's mostly a page number
    - ASCII Roman: IVXLCDM - Unicode Roman: U+2160–U+216F, U+2170–U+217F
    """
    t = re.sub(r"[\s,.\-–—/\\:;·•…()［］\[\]{}]", "", s)
    if not t:
        return False
    if re.fullmatch(r"[0-9０-９]+", t):
        return True
    if re.fullmatch(r"(?i)[IVXLCDM]+", t):
        return True
    if re.fullmatch(r"[\u2160-\u216F\u2170-\u217F]+", t):
        return True
    return False

def load_labeled_chunk(path: Path, lang: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*<([^>]+)>\s*(.*)$", line)
        if not m:
            continue
        lab, txt = m.group(1), m.group(2).strip()
        if not txt:
            continue
        for s in sent_split(txt, lang):
            out.append((lab, s))
    return out

# merge "Text" label texts (except for number lines)
def merge_text_only(items: List[Tuple[str, str]], lang: str) -> List[str]:
    """
    Concat <Text> label texts and split
    BEFORE <Text> ~. I run every <Page-header> xxx <Text> morning. ~
    AFTER <Text> ~. <Text>I run every morning. <Page-header> xxx <Text> ~
    """
    out: List[str] = []
    buf: str | None = None
    pending_nontext: List[str] = []
    sep = " " if lang != "ja" else ""

    for lab, raw in items:
        if lab == "Text":
            subs = sent_split(raw, lang)
            if not subs and raw.strip():
                subs = [raw.strip()]

            for sent in subs:
                if buf is None:
                    buf = sent
                else:
                    can_merge = (not ends_with_terminal(buf, lang)) \
                                and (not is_numeric_or_roman_only(buf)) \
                                and (not is_numeric_or_roman_only(sent))
                    if can_merge:
                        buf = (buf + sep + sent).strip()
                    else:
                        out.append(buf)
                        if pending_nontext:
                            out.extend(pending_nontext)
                            pending_nontext = []
                        buf = sent

                if buf is not None and ends_with_terminal(buf, lang) and pending_nontext:
                    out.append(buf)
                    out.extend(pending_nontext)
                    pending_nontext = []
                    buf = None

        else:
            if buf is not None and (not ends_with_terminal(buf, lang)) and (not is_numeric_or_roman_only(buf)):
                pending_nontext.append(raw)
            else:
                if buf is not None:
                    out.append(buf)
                    buf = None
                if pending_nontext:
                    out.extend(pending_nontext)
                    pending_nontext = []
                out.append(raw)

    if buf is not None:
        out.append(buf)
    if pending_nontext:
        out.extend(pending_nontext)

    return [s for s in (x.strip() for x in out) if s]

# split segment texts into sentences
def collect_per_chunk_sentences(pairs_dir: Path, lang: str) -> List[List[str]]:
    paths = sorted(
        pairs_dir.glob(f"chunk*-label-{lang}.txt"),
        key=lambda p: int(re.search(r"chunk(\d+)-", p.name).group(1)) if re.search(r"chunk(\d+)-", p.name) else 10**9
    )
    per_chunk: List[List[str]] = []
    for p in paths:
        labeled = load_labeled_chunk(p, lang)
        merged  = merge_text_only(labeled, lang)
        per_chunk.append(merged)
    return per_chunk

# write a file: segment -> sentences
def write_chunk_sentence_files(per_chunk: List[List[str]], out_dir: Path, prefix: str = "chunk", ext: str = ".txt") -> None:
    for i, sents in enumerate(per_chunk, start=1):
        (out_dir / f"{prefix}{i}{ext}").write_text("\n".join(sents), encoding="utf-8")


# main
def run_sentence_alignment(src_lang, tgt_lang, align_method, model_emb, model_nllb, skip_mt=False):
    
    in_chunks_dir = Path("result/chunks_aligned")
    out_sents_dir = Path("result/sentences_aligned")
    if out_sents_dir.exists():
        shutil.rmtree(out_sents_dir)
    out_sents_dir.mkdir(parents=True, exist_ok=True)

    # split segment-unit texts into sentences
    en_per_chunk = collect_per_chunk_sentences(in_chunks_dir, src_lang)
    ja_per_chunk = collect_per_chunk_sentences(in_chunks_dir, tgt_lang)

    src_dir = out_sents_dir / f"source({src_lang})"
    tgt_dir = out_sents_dir / f"target({tgt_lang})"
    src_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    # write down in files（chunk1.txt, chunk2.txt, ...）
    write_chunk_sentence_files(en_per_chunk, src_dir, prefix="chunk", ext=".txt")
    write_chunk_sentence_files(ja_per_chunk, tgt_dir, prefix="chunk", ext=".txt")

    # machine translate(NLLB) -> tokenize -> Bleualign -> merge
    if align_method == "bleualign":
        run_bleualign(src_dir, tgt_dir, out_sents_dir, src_lang=src_lang, tgt_lang=tgt_lang,
                    mt="nllb", model_nllb=model_nllb, skip_mt=skip_mt)
    
    elif align_method == "vecalign":
        run_vecalign(src_dir, tgt_dir, out_sents_dir, src_lang=src_lang, tgt_lang=tgt_lang,
                     model_emb=model_emb)