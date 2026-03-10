"""
index_utils.py v6 — Agentic RAG
=============================
Semantic Search (MiniLM) + Vector Database (FAISS) + Reranker (FlashRank).
Multi-producto soportado de forma nativa.
"""

import math
import re
import pickle
import numpy as np
from pathlib import Path

# Nuevas dependencias para RAG Agéntico Semántico
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from flashrank import Ranker, RerankRequest
except ImportError:
    print("Faltan dependencias. Debes instalar:")
    print("pip install sentence-transformers faiss-cpu flashrank")


def normalize(text: str) -> str:
    """Conserva el normalizado básico por si se necesita para metadata."""
    return (text.lower()
            .replace('á','a').replace('é','e').replace('í','i')
            .replace('ó','o').replace('ú','u').replace('ñ','n'))


def _deduplicate(results: list[dict], threshold: float = 0.85) -> list[dict]:
    """Elimina fragmentos demasiado similares usando Jaccard en palabras."""
    seen: list[set] = []
    out = []
    for r in results:
        tokens = set(r['content'].lower().split())
        if not any(tokens and prev and len(tokens & prev) / len(tokens | prev) > threshold
                   for prev in seen):
            out.append(r)
            seen.append(tokens)
    return out


class SemanticIndex:
    """Índice semántico usando MiniLM, FAISS y FlashRank."""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.documents: list[dict] = []
        self.products: list[str] = []
        self.embedding_model_name = embedding_model_name
        self.encoder = None
        self.faiss_index = None
        self.reranker = None
        self.is_loaded = False

    def _lazy_init(self):
        """Inicializa los modelos en memoria solo cuando se van a usar."""
        if not self.is_loaded:
            print(f"Cargando modelo de embeddings ({self.embedding_model_name})...")
            self.encoder = SentenceTransformer(self.embedding_model_name)
            print("Cargando Reranker (FlashRank)...")
            self.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./output/flashrank_cache")
            self.is_loaded = True

    def build(self, chunks: list[dict], tables: list[dict]) -> None:
        """Construye la base de datos vectorial a partir de datos estructurados."""
        self._lazy_init()
        print("Construyendo índice Semántico v6...")
        self.documents = []

        # 1. Preparar Documentos Texto
        for chunk in chunks:
            self.documents.append({
                "id": chunk["id"], 
                "type": "text",
                "product": chunk.get("product", "UNKNOWN"),
                "pdf": chunk.get("pdf", ""),
                "page": chunk["page"], 
                "section": chunk["section"],
                "content": chunk["text"],
            })

        # 2. Preparar Documentos Tablas (Serializadas)
        for table in tables:
            txt = f"Producto: {table.get('product','')}. Seccion: {table['section']}. "
            txt += "Columnas: " + ", ".join(table["headers"]) + ". "
            for row in table["rows"]:
                txt += " | ".join(f"{k}: {v}" for k, v in row.items() if v) + ". "
            self.documents.append({
                "id": table["id"], 
                "type": "table",
                "product": table.get("product", "UNKNOWN"),
                "pdf": table.get("pdf", ""),
                "page": table["page"], 
                "section": table["section"],
                "content": txt, 
                "headers": table.get("headers", []),
                "rows": table.get("rows", [])
            })

        self.products = sorted(set(d["product"] for d in self.documents))
        n_docs = len(self.documents)

        # 3. Calcular Embeddings en bloque
        print(f"Generando embeddings para {n_docs} documentos...")
        texts = [doc["content"] for doc in self.documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # 4. Construir FAISS Index (Inner Product para similitud coseno)
        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)  # L2 norm para que IP actúe como Cosine Similarity
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

        print(f"  {n_docs} docs indexados | dimensiones: {dim} | productos: {', '.join(self.products)}")


    def search(self, query: str, top_k: int = 6, product_filter: str = None) -> list[dict]:
        """Busca y re-ordena documentos basándose puramente en semántica profunda."""
        self._lazy_init()
        if not query.strip() or self.faiss_index is None:
            return []

        # 1. Embedding de la consulta
        q_emb = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        # 2. Recuperación Densa Amplia (Buscamos más documentos de los necesarios para luego filtrar y reranquear)
        retrieve_k = top_k * 4 
        distances, indices = self.faiss_index.search(q_emb, retrieve_k)
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            doc = self.documents[idx]
            if product_filter and doc["product"] != product_filter:
                continue
            doc_copy = doc.copy()
            doc_copy["faiss_score"] = float(distances[0][i])
            candidates.append(doc_copy)

        if not candidates:
            return []

        # 3. Reranking cruzado usando FlashRank
        # Preparar formato para FlashRank
        passages = []
        for doc in candidates:
            passages.append({
                "id": doc["id"],
                "text": doc["content"],
                "meta": doc
            })
            
        rerankrequest = RerankRequest(query=query, passages=passages)
        rerank_results = self.reranker.rerank(rerankrequest)

        # 4. Ensamblar resultados finales
        final_results = []
        for res in rerank_results:
            orig_doc = res["meta"]
            orig_doc["score"] = res["score"] # Score de FlashRank
            final_results.append(orig_doc)

        return _deduplicate(final_results)[:top_k]


    def search_with_stats(self, query: str, top_k: int = 6, product_filter: str = None) -> dict:
        results = self.search(query, top_k=top_k, product_filter=product_filter)
        return {
            "results": results, 
            "query_tokens": len(query.split()),
            "expanded": False,
            "expanded_query": None,
            "total_docs": len(self.documents),
            "products_found": list({r["product"] for r in results}),
        }

    def save(self, path: Path) -> None:
        """Guarda FAISS y metadata por separado para evitar problemas de Pickling."""
        output_dir = path.parent
        faiss_path = output_dir / "index.faiss"
        meta_path = output_dir / "metadata.pkl"
        
        # Guarda el index en formato binario de FAISS
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(faiss_path))
            
        # Guarda los documentos y metadata en formato pickle
        meta_data = {
            "documents": self.documents,
            "products": self.products,
            "embedding_model_name": self.embedding_model_name
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta_data, f)
            
        # También mantiene un archivo dummy en index.pkl para no quebrar scripts viejos inmediatamente
        with open(path, "wb") as f:
            pickle.dump({"warning": "Index v6 is split in metadata.pkl and index.faiss"}, f)
            
        print(f"Índice Semántico guardado: {faiss_path} y {meta_path}")

    @classmethod
    def load(cls, path: Path) -> "SemanticIndex":
        """Reconstruye el objeto desde FAISS + Metadata."""
        output_dir = path.parent
        faiss_path = output_dir / "index.faiss"
        meta_path = output_dir / "metadata.pkl"
        
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No se encontraron los archivos {faiss_path} o {meta_path}. Debes reconstruir el índice (paso 2).")
            
        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)
            
        instance = cls(embedding_model_name=meta_data.get("embedding_model_name", 'all-MiniLM-L6-v2'))
        instance.documents = meta_data["documents"]
        instance.products = meta_data["products"]
        
        instance.faiss_index = faiss.read_index(str(faiss_path))
        print("Índice semántico cargado correctamente.")
        return instance