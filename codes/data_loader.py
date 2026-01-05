from pathlib import Path
import shutil
import fitz

def load_pages(pdf_path, out_dir, dpi=300):
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    doc = fitz.open(str(pdf_path))
    zoom = float(dpi) / 72.0
    mat = fitz.Matrix(zoom, zoom)

    out = []
    for idx in range(doc.page_count):
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = out_dir / f"{pdf_path.stem}_page{idx+1:03}.jpg"
        pix.save(str(img_path))  # format inferred from extension
        out.append(img_path)

        # help GC / release resources
        del page, pix

    doc.close()
    return out


def pdf_to_jpg_folder(pdf_dir, output_dir, dpi=300, langs=("en", "ja")):
    """
    Convert PDFs under pdf_dir/{en,ja} to JPGs (streaming).
    """
    base_dir = Path(pdf_dir)
    out_base = Path(output_dir)
    total = 0
    
    for lang in (langs or ("en", "ja")):
        pdf_files = sorted((base_dir / lang).glob("*.pdf"))
        out_dir = out_base / lang
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for pdf_file in pdf_files:
            jpg_list = load_pages(pdf_file, out_dir, dpi=dpi)
            print(f"[{lang}] {pdf_file.name} → {len(jpg_list)} pages saved in {out_dir}")
            total += len(jpg_list)

    return total