from pathlib import Path
import fitz  # pymupdf

pdf_dir = Path("./")
out_dir = Path("./")
out_dir.mkdir(parents=True, exist_ok=True)

for pdf_path in pdf_dir.glob("*.pdf"):
    doc = fitz.open(pdf_path)
    texts = []

    for page_idx, page in enumerate(doc, start=1):
        text = page.get_text("text")
        texts.append(f"\n\n[Page {page_idx}]\n{text}")

    out_path = out_dir / f"{pdf_path.stem}.txt"
    out_path.write_text("\n".join(texts), encoding="utf-8")

    print(f"converted: {pdf_path.name} -> {out_path}")