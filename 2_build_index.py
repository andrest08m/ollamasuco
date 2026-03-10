"""
PASO 2: CONSTRUIR ÍNDICE v6 Semántico
=====================================
Lee chunks.json + tables.json y construye el índice embedding multi-producto (FAISS).

Uso:
    python 2_build_index.py
"""

import json
from pathlib import Path
from index_utils import SemanticIndex

OUTPUT_DIR = Path("output")
INDEX_FILE = OUTPUT_DIR / "index.pkl"  # Compatibilidad, creará faiss local

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

    index = SemanticIndex()
    index.build(chunks, tables)
    index.save(INDEX_FILE)

    print(f"\nÍndice listo en: {OUTPUT_DIR}/")
    print("Ejecuta: python 3_chat.py")

if __name__ == "__main__":
    main()