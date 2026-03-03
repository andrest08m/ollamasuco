"""
index_utils.py v5
=================
BM25 + Reranker multi-producto.
"""

import math
import re
import pickle
from collections import Counter, defaultdict
from pathlib import Path


STOPWORDS = {
    'de','la','el','en','y','a','que','se','del','los','las','un','una',
    'con','por','para','es','al','lo','su','no','si','o','como','pero',
    'sus','le','ya','fue','son','este','esta','esto','estos','estas',
    'cuando','cada','vez','puede','pueden','debe','deben','entre','sobre',
    'the','of','and','to','in','is','it','for','on','are','this','that',
    'with','be','not','or','an','at','from','also',
}

HIGH_PRIORITY_SECTIONS = {
    'especificaciones','medicion','medición','rango','funcion','función',
    'caracteristicas','características','seguridad','bluetooth','comunicacion',
    'descripcion','descripción','comparador','filtro','aplicacion','uso','producto',
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r'([a-záéíóúüñ])([A-ZÁÉÍÓÚÜ])', r'\1 \2', text)
    text = text.lower()
    tokens = re.findall(r'[a-záéíóúüñ0-9]{2,}', text)
    return [t for t in tokens if t not in STOPWORDS]


def normalize(text: str) -> str:
    return (text.lower()
            .replace('á','a').replace('é','e').replace('í','i')
            .replace('ó','o').replace('ú','u').replace('ñ','n'))


def expand_query(query: str) -> str:
    q = normalize(query)
    expansions = []

    if any(w in q for w in ['tipo','que es','descripcion','modelo','para que sirve','finalidad','pinza','producto']):
        expansions += ['descripcion producto tipo instrumento medir aplicacion uso']
    if any(w in q for w in ['rango','corriente','medir','medicion','amperio','amperaje',
                              'minimo','minima','maximo','maxima','limite','capacidad','ma ','miliamperio','fuga','sobrecarga']):
        expansions += ['rango medicion corriente mA amperios minimo maximo especificaciones']
    if any(w in q for w in ['especificacion','tecnico','tecnica','dato','caracteristica','precision','resolucion']):
        expansions += ['especificaciones tecnicas rango precision resolucion']
    if any(w in q for w in ['cat','categoria','voltaje','seguridad','clasificacion','norma','iec']):
        expansions += ['categoria medicion CAT voltaje clasificacion seguridad']
    if any(w in q for w in ['comparador','umbral','alarma','alerta','pitido','aviso','comp']):
        expansions += ['funcion comparador umbral pitido advertencia']
    if any(w in q for w in ['bluetooth','inalambric','wifi','wireless','z3210','gennect','app','conectar']):
        expansions += ['comunicacion inalambrica bluetooth adaptador Z3210 GENNECT']
    if any(w in q for w in ['pila','bateria','alimentacion','aaa','energia']):
        expansions += ['pilas bateria tipo alimentacion AAA']
    if any(w in q for w in ['hold','retener','retencion','congelar','pausar']):
        expansions += ['funcion retencion HOLD congelar lectura']
    if any(w in q for w in ['filtro','filter','frecuencia','inversor','ruido','hz']):
        expansions += ['funcion filtro paso bajo frecuencia inversor']
    if any(w in q for w in ['peso','dimension','tamano','largo','ancho','alto','gramo','mm']):
        expansions += ['peso dimensiones mm gramos especificaciones fisicas']
    if any(w in q for w in ['mordaza','apertura','diametro','cable','conductor']):
        expansions += ['diametro maximo mordaza apertura mm conductor']
    if any(w in q for w in ['recomendar','para quien','para que persona','util','sirve','uso',
                              'ventaja','cliente','vender','comprar','elegir','cual es mejor']):
        expansions += ['descripcion producto aplicacion uso ventajas caracteristicas principales']
    if any(w in q for w in ['problema','fallo','falla','detectar','identificar','localizar',
                              'verificar','comprobar','inspeccionar','revisar']):
        expansions += ['medicion corriente fuga sobrecarga aplicacion inspeccion verificacion']
    if any(w in q for w in ['diferencia','comparar','vs','versus','mejor','cual','entre',
                              'alternativa','opcion','similar']):
        expansions += ['descripcion producto especificaciones rango capacidad aplicacion']

    return query + ' ' + ' '.join(expansions) if expansions else query


def _section_boost(section: str, query_tokens: list[str]) -> float:
    section_lower = section.lower()
    boost = sum(0.4 for w in HIGH_PRIORITY_SECTIONS if w in section_lower)
    boost += sum(0.7 for t in query_tokens if len(t) > 3 and t in section_lower)
    return min(boost, 2.5)


def _exact_phrase_boost(content: str, query: str) -> float:
    words = query.lower().split()
    c = content.lower()
    score = sum(1.5 for i in range(len(words)-1)
                if len(' '.join(words[i:i+3])) > 6 and ' '.join(words[i:i+3]) in c)
    return min(score, 3.0)


def _deduplicate(results: list[dict], threshold: float = 0.65) -> list[dict]:
    seen: list[set] = []
    out = []
    for r in results:
        tokens = set(tokenize(r['content']))
        if not any(tokens and prev and len(tokens & prev) / len(tokens | prev) > threshold
                   for prev in seen):
            out.append(r)
            seen.append(tokens)
    return out


class TFIDFIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.documents: list[dict] = []
        self.tfidf_matrix: list[dict] = []
        self.idf: dict[str, float] = {}
        self.k1 = k1
        self.b = b
        self.avg_doc_len: float = 0.0
        self.products: list[str] = []

    def build(self, chunks: list[dict], tables: list[dict]) -> None:
        print("Construyendo índice BM25 v5...")
        self.documents = []

        for chunk in chunks:
            self.documents.append({
                "id": chunk["id"], "type": "text",
                "product": chunk.get("product", "UNKNOWN"),
                "pdf": chunk.get("pdf", ""),
                "page": chunk["page"], "section": chunk["section"],
                "content": chunk["text"],
                "tokens": tokenize(chunk["text"])
            })

        for table in tables:
            txt = f"Producto: {table.get('product','')}. Seccion: {table['section']}. "
            txt += "Columnas: " + ", ".join(table["headers"]) + ". "
            for row in table["rows"]:
                txt += " | ".join(f"{k}: {v}" for k, v in row.items() if v) + ". "
            self.documents.append({
                "id": table["id"], "type": "table",
                "product": table.get("product", "UNKNOWN"),
                "pdf": table.get("pdf", ""),
                "page": table["page"], "section": table["section"],
                "content": txt, "headers": table.get("headers", []),
                "rows": table.get("rows", []), "tokens": tokenize(txt)
            })

        self.products = sorted(set(d["product"] for d in self.documents))
        n_docs = len(self.documents)
        total_tokens = sum(len(d["tokens"]) for d in self.documents)
        self.avg_doc_len = total_tokens / n_docs if n_docs else 1.0

        doc_freq: dict[str, int] = defaultdict(int)
        for doc in self.documents:
            for t in set(doc["tokens"]):
                doc_freq[t] += 1

        self.idf = {t: math.log((n_docs - f + 0.5) / (f + 0.5) + 1)
                    for t, f in doc_freq.items()}

        self.tfidf_matrix = []
        for doc in self.documents:
            tf = Counter(doc["tokens"])
            dl = len(doc["tokens"]) or 1
            bm25 = {}
            for term, count in tf.items():
                idf = self.idf.get(term, 0)
                num = count * (self.k1 + 1)
                den = count + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
                bm25[term] = idf * num / den
            self.tfidf_matrix.append(bm25)

        print(f"  {n_docs} docs | {len(self.idf)} términos | productos: {', '.join(self.products)}")

    def _bm25_score(self, idx: int, query_tokens: list[str]) -> float:
        return sum(self.tfidf_matrix[idx].get(t, 0.0) for t in query_tokens)

    def search(self, query: str, top_k: int = 6, product_filter: str = None) -> list[dict]:
        expanded = expand_query(query)
        qt = tokenize(expanded)
        if not qt:
            return []

        candidates = []
        for idx, bm25 in enumerate(self.tfidf_matrix):
            doc = self.documents[idx]
            if product_filter and doc["product"] != product_filter:
                continue
            score = sum(bm25.get(t, 0.0) for t in qt)
            if score > 0:
                candidates.append((idx, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for idx, bs in candidates[:top_k * 4]:
            doc = self.documents[idx]
            final = bs + _section_boost(doc['section'], qt) + \
                    _exact_phrase_boost(doc['content'], query) + \
                    (0.5 if doc['type'] == 'table' else 0.0)
            reranked.append((idx, bs, final))

        reranked.sort(key=lambda x: x[2], reverse=True)

        results = []
        for idx, bs, fs in reranked[:top_k * 2]:
            doc = self.documents[idx].copy()
            doc["score"] = round(fs, 4)
            doc["score_bm25"] = round(bs, 4)
            doc.pop("tokens", None)
            results.append(doc)

        return _deduplicate(results)[:top_k]

    def search_multi_product(self, query: str, top_k_per_product: int = 3) -> list[dict]:
        """Trae los mejores chunks de CADA producto para comparaciones."""
        all_results = []
        for product in self.products:
            results = self.search(query, top_k=top_k_per_product, product_filter=product)
            all_results.extend(results)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return _deduplicate(all_results)

    def search_with_stats(self, query: str, top_k: int = 6,
                           product_filter: str = None, multi_product: bool = False) -> dict:
        expanded = expand_query(query)
        qt = tokenize(expanded)
        if multi_product:
            results = self.search_multi_product(query, top_k_per_product=3)
        else:
            results = self.search(query, top_k=top_k, product_filter=product_filter)
        return {
            "results": results, "query_tokens": qt,
            "expanded": expanded != query,
            "expanded_query": expanded if expanded != query else None,
            "total_docs": len(self.documents),
            "products_found": list({r["product"] for r in results}),
        }

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Índice guardado: {path}")

    @classmethod
    def load(cls, path: Path) -> "TFIDFIndex":
        with open(path, "rb") as f:
            return pickle.load(f)