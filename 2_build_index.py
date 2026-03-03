"""
PASO 2: CONSTRUIR ÍNDICE v5
============================
Lee chunks.json + tables.json y construye el índice BM25 multi-producto.

Uso:
    python 2_build_index.py
"""

import json
from pathlib import Path
from index_utils import TFIDFIndex

OUTPUT_DIR = Path("output")
INDEX_FILE = OUTPUT_DIR / "index.pkl"

def main():
    chunks_file = OUTPUT_DIR / "chunks.json"
    tables_file = OUTPUT_DIR / "tables.json"

    if not chunks_file.exists():
        print("ERROR: No existe output/chunks.json")
        print("Ejecuta primero: python 1_extract_pdf.py manuales/")
        return

    print("Cargando datos...")
    with open(chunks_file, encoding="utf-8") as f:
        chunks = json.load(f)
    with open(tables_file, encoding="utf-8") as f:
        tables = json.load(f)

    print(f"  {len(chunks)} chunks, {len(tables)} tablas")

    index = TFIDFIndex()
    index.build(chunks, tables)
    index.save(INDEX_FILE)

    print(f"\nÍndice listo: {INDEX_FILE}")
    print("Ejecuta: streamlit run app_streamlit.py")

if __name__ == "__main__":
    main()