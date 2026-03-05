"""
PASO 1: EXTRACTOR MULTI-PDF v5
================================
Procesa TODOS los PDFs de una carpeta.
Cada chunk lleva metadato del producto de origen.

Uso:
    python 1_extract_pdf.py                      # procesa carpeta manuales/
    python 1_extract_pdf.py manuales/            # carpeta especifica
    python 1_extract_pdf.py manuales/CM4001.pdf  # un solo PDF

Salida:
    output/chunks.json
    output/tables.json
    output/products.json
"""

import json
import re
import sys
from pathlib import Path

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 120
MIN_CHUNK_LEN = 80
HEADER_NOISE  = re.compile(r'HIOKI\s+\S+|\b\d{1,3}\s*$', re.MULTILINE)

try:
    import pymupdf4llm
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_product_name(pdf_path: Path) -> str:
    """Extrae nombre del producto del nombre del archivo o primera página."""
    stem = pdf_path.stem.upper()
    # Patrones comunes HIOKI: CM4001, PW3337, DT4282, FT6380, CM4371-50, etc.
    # Evita tomar la 'A' o 'B' de los códigos del manual (ej. A966-01)
    m = re.search(r'([A-Z]{2,4}\d{3,5}(?:-\d{2})?)', stem)
    if m:
        return m.group(1)
    return stem[:20]


def clean_text(text: str) -> str:
    text = HEADER_NOISE.sub('', text)
    lines = []
    for line in text.split('\n'):
        # Filtrar líneas de índice (muchos puntos consecutivos o espacios)
        if re.search(r'\.{5,}', line):
            continue
        line = re.sub(r'[ \t]{2,}', ' ', line).strip()
        if line:
            lines.append(line)
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_junk_chunk(body: str) -> bool:
    """Detecta si un chunk es texto basura como un disclaimer legal o un índice residual."""
    lower_body = body.lower()
    
    # 1. Disclaimers o legales repetitivos
    disclaimers = [
        "contenido del manual está sujeto a cambios",
        "puede descargar la versión más reciente",
        "marcas registradas y nombres comerciales",
        "excel es una marca",
        "palabra bluetooth",
        "registro de productos",
        "este manual se ha escrito para"
    ]
    for d in disclaimers:
        if d in lower_body:
            return True
            
    # 2. Índice residual
    if lower_body.count('.....') > 1:
        return True
        
    return False


def detect_section_title(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if re.match(r'^#{1,3}\s+\S', line):
        return True
    if not (4 <= len(line) <= 90):
        return False
    if line.endswith('.') or line.endswith(','):
        return False
    if re.match(r'^\d+\s*$', line):
        return False
    if not (line[0].isupper() or line[0] in '#¿'):
        return False
    return True


def text_to_chunks(text: str, page_num: int, product: str, pdf_name: str) -> list[dict]:
    chunks = []
    lines = text.split('\n')
    current_section = "General"
    current_text = ""
    chunk_idx = 0

    def save_chunk():
        nonlocal chunk_idx, current_text
        body = current_text.strip()
        
        if len(body) >= MIN_CHUNK_LEN and not is_junk_chunk(body):
            chunks.append({
                "id":         f"{product}_p{page_num}_{chunk_idx}",
                "product":    product,
                "pdf":        pdf_name,
                "page":       page_num,
                "section":    current_section,
                "text":       body,
                "char_count": len(body),
            })
            chunk_idx += 1
        words = body.split()
        overlap = words[-20:] if len(words) > 20 else words
        current_text = " ".join(overlap) + " "

    for line in lines:
        if detect_section_title(line):
            if current_text.strip():
                save_chunk()
            current_section = re.sub(r'^#+\s+', '', line).strip()
            current_text = line.strip() + "\n"
        else:
            current_text += line + "\n"
            if len(current_text) >= CHUNK_SIZE:
                save_chunk()

    if current_text.strip():
        save_chunk()

    return chunks


def parse_markdown_table(table_md: str, page_num: int, section: str,
                          t_idx: int, product: str, pdf_name: str) -> dict | None:
    lines = [l.strip() for l in table_md.strip().split('\n') if l.strip()]
    data_lines = [l for l in lines if not re.match(r'^\|[\s\-|:]+\|$', l)]
    if len(data_lines) < 2:
        return None

    def split_row(line):
        return [cell.strip() for cell in re.split(r'\|', line) if cell.strip()]

    headers = split_row(data_lines[0])
    rows = []
    for row_line in data_lines[1:]:
        cells = split_row(row_line)
        if not cells:
            continue
        row_dict = {headers[i] if i < len(headers) else f"col_{i}": cell
                    for i, cell in enumerate(cells)}
        if any(v.strip() for v in row_dict.values()):
            rows.append(row_dict)

    if not rows:
        return None

    return {
        "id":        f"table_{product}_p{page_num}_{t_idx}",
        "product":   product,
        "pdf":       pdf_name,
        "page":      page_num,
        "section":   section,
        "headers":   headers,
        "rows":      rows,
        "row_count": len(rows),
    }


# ── Extracción ─────────────────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> tuple[list[dict], list[dict], dict]:
    product  = extract_product_name(pdf_path)
    pdf_name = pdf_path.name
    print(f"\n  [{product}] {pdf_name}")

    all_chunks: list[dict] = []
    all_tables: list[dict] = []
    current_section = "General"

    if HAS_PYMUPDF:
        try:
            pages_data = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
            if not isinstance(pages_data, list):
                raise ValueError("page_chunks no soportado")

            for page_entry in pages_data:
                raw_text = page_entry.get("text", "") if isinstance(page_entry, dict) else str(page_entry)
                meta     = page_entry.get("metadata", {}) if isinstance(page_entry, dict) else {}
                page_num = meta.get("page", 0) + 1

                cleaned = clean_text(raw_text)
                if not cleaned:
                    continue

                for t_idx, tmd in enumerate(re.findall(r'(\|.+\|(?:\n\|.+\|)+)', cleaned, re.MULTILINE)):
                    td = parse_markdown_table(tmd, page_num, current_section, t_idx, product, pdf_name)
                    if td:
                        all_tables.append(td)

                for line in cleaned.split('\n'):
                    if detect_section_title(line.strip()):
                        current_section = re.sub(r'^#+\s+', '', line).strip()

                all_chunks.extend(text_to_chunks(cleaned, page_num, product, pdf_name))

            print(f"    pymupdf4llm: {len(all_chunks)} chunks, {len(all_tables)} tablas")
        except Exception as e:
            print(f"    pymupdf4llm falló ({e}), usando pdfminer...")
            all_chunks, all_tables = _extract_pdfminer(pdf_path, product, pdf_name)
    else:
        all_chunks, all_tables = _extract_pdfminer(pdf_path, product, pdf_name)

    product_info = {
        "product":    product,
        "pdf":        pdf_name,
        "chunks":     len(all_chunks),
        "tables":     len(all_tables),
        "sections":   list({c["section"] for c in all_chunks})[:20],
    }

    return all_chunks, all_tables, product_info


def _extract_pdfminer(pdf_path: Path, product: str, pdf_name: str) -> tuple[list[dict], list[dict]]:
    import pdfplumber
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams

    full_text = extract_text(str(pdf_path), laparams=LAParams(
        word_margin=0.1, char_margin=2.0, line_margin=0.5, boxes_flow=0.5))

    raw_pages = full_text.split('\x0c')
    all_chunks = []
    page_sections = {}
    current_section = "General"

    for page_num, raw_text in enumerate(raw_pages, start=1):
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        for line in cleaned.split('\n'):
            if detect_section_title(line.strip()):
                current_section = line.strip()
        page_sections[page_num] = current_section
        all_chunks.extend(text_to_chunks(cleaned, page_num, product, pdf_name))

    all_tables = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            section = page_sections.get(page_num, "General")
            for t_idx, table in enumerate(page.extract_tables() or []):
                if not table or len(table) < 2:
                    continue
                headers = [re.sub(r'\s+', ' ', str(c or "")).strip() for c in table[0]]
                rows = []
                for row in table[1:]:
                    rd = {headers[i] if i < len(headers) else f"col_{i}": str(c or "").strip()
                          for i, c in enumerate(row)}
                    if any(v.strip() for v in rd.values()):
                        rows.append(rd)
                if rows:
                    all_tables.append({
                        "id": f"table_{product}_p{page_num}_{t_idx}",
                        "product": product, "pdf": pdf_name,
                        "page": page_num, "section": section,
                        "headers": headers, "rows": rows, "row_count": len(rows),
                    })

    print(f"    pdfminer: {len(all_chunks)} chunks, {len(all_tables)} tablas")
    return all_chunks, all_tables


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Determinar qué procesar
    if len(sys.argv) >= 2:
        target = Path(sys.argv[1])
    else:
        target = Path("manuales")

    if target.is_file() and target.suffix.lower() == ".pdf":
        pdf_files = [target]
    elif target.is_dir():
        pdf_files = sorted(target.glob("**/*.pdf"))
    else:
        print(f"No se encontró: {target}")
        sys.exit(1)

    if not pdf_files:
        print(f"No hay PDFs en: {target}")
        sys.exit(1)

    print(f"PDFs encontrados: {len(pdf_files)}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Cargar chunks existentes para modo incremental
    existing_chunks_file = output_dir / "chunks.json"
    existing_tables_file = output_dir / "tables.json"
    existing_products_file = output_dir / "products.json"

    all_chunks: list[dict] = []
    all_tables: list[dict] = []
    all_products: dict = {}

    if existing_products_file.exists():
        with open(existing_products_file, encoding="utf-8") as f:
            all_products = json.load(f)
        # Preguntar si re-procesar o agregar
        existing_pdfs = {p["pdf"] for p in all_products.values()}
        new_pdfs = [f for f in pdf_files if f.name not in existing_pdfs]
        already = [f for f in pdf_files if f.name in existing_pdfs]

        if already and new_pdfs:
            print(f"\nYa indexados: {len(already)} PDFs")
            print(f"Nuevos a agregar: {len(new_pdfs)} PDFs")
            # Cargar existentes
            with open(existing_chunks_file, encoding="utf-8") as f:
                all_chunks = json.load(f)
            with open(existing_tables_file, encoding="utf-8") as f:
                all_tables = json.load(f)
            pdf_files = new_pdfs
        elif not new_pdfs:
            print("\nTodos los PDFs ya están indexados. Usa --force para re-indexar.")
            if "--force" not in sys.argv:
                sys.exit(0)

    # Procesar PDFs
    for pdf_path in pdf_files:
        try:
            chunks, tables, product_info = extract_pdf(pdf_path)
            all_chunks.extend(chunks)
            all_tables.extend(tables)
            all_products[product_info["product"]] = product_info
        except Exception as e:
            print(f"  ERROR procesando {pdf_path.name}: {e}")

    # Guardar
    with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    with open(output_dir / "tables.json", "w", encoding="utf-8") as f:
        json.dump(all_tables, f, ensure_ascii=False, indent=2)

    with open(output_dir / "products.json", "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"RESUMEN:")
    print(f"  Productos indexados : {len(all_products)}")
    print(f"  Total chunks        : {len(all_chunks)}")
    print(f"  Total tablas        : {len(all_tables)}")
    print(f"\nProductos:")
    for prod, info in all_products.items():
        print(f"  {prod:12} → {info['chunks']} chunks, {info['tables']} tablas")
    print(f"\nGuardado en: {output_dir}/")


if __name__ == "__main__":
    main()