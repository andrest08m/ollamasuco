"""
PASO 3: CHAT v3 — RAG anti-alucinacion con Ollama
==================================================
Uso:
    python 3_chat.py
    python 3_chat.py --model llama3.2
    python 3_chat.py --debug       <- muestra contexto enviado al LLM
    python 3_chat.py --extract     <- extrae datos estructurados
"""

import argparse
import json
import pickle
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

from index_utils import TFIDFIndex   # necesario para deserializar pickle


OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"
INDEX_FILE    = Path("output/index.pkl")
TOP_K_CONTEXT = 6           # mas contexto = mejor cobertura
MAX_CONTEXT_CHARS = 4000


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# Disenado para:
#   - Forzar al modelo a usar SOLO el contexto dado
#   - Responder con datos exactos del manual
#   - No inventar cuando no sabe
#   - Manejar preguntas vagas expandiendolas el mismo
SYSTEM_PROMPT = """Eres un asistente tecnico especializado en manuales de instrumentos HIOKI.

Tu unica fuente de informacion es el texto que aparece dentro del bloque
[CONTEXTO DEL MANUAL] en cada mensaje. Ese texto fue extraido directamente del manual oficial.

REGLAS OBLIGATORIAS:

REGLA 1 — RESPONDE SOLO CON EL CONTEXTO
Usa exclusivamente la informacion del bloque [CONTEXTO DEL MANUAL].
No uses conocimiento externo. No supongas ni completes con logica propia.

REGLA 2 — SI LA RESPUESTA ESTA EN EL CONTEXTO, RESPONDELA
Lee el contexto completo antes de responder. Si la informacion esta ahi,
aunque sea parcialmente, respondela con los datos exactos que aparecen.

REGLA 3 — SI NO ESTA EN EL CONTEXTO, DI EXACTAMENTE ESTO
"Esta informacion no aparece en el fragmento del manual disponible."
No digas "probablemente", "generalmente" ni inventes datos.

REGLA 4 — DATOS TECNICOS: CITA VALORES EXACTOS
Para rangos, voltajes, dimensiones, categorias: copia el valor EXACTO del contexto.
Ejemplo correcto: "mide desde 0,60 mA hasta 600,0 A"
Ejemplo incorrecto: "mide corrientes pequeñas hasta grandes"

REGLA 5 — RESPONDE EN EL IDIOMA DE LA PREGUNTA

REGLA 6 — SE DIRECTO
Primero da la respuesta. Luego el detalle si es necesario. Sin introducciones."""


def check_ollama(model: str) -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if not any(model in m for m in models):
                print(f"Modelo '{model}' no encontrado. Disponibles: {models}")
                print(f"Instala con: ollama pull {model}")
                return False
            return True
    except Exception as e:
        print(f"Ollama no responde: {e}")
        print("Inicia con: ollama serve")
        return False


def query_ollama(prompt: str, model: str, stream: bool = True) -> str:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature":    0.0,   # deterministico - maximo rigor
            "top_p":          1.0,
            "num_predict":    512,
            "repeat_penalty": 1.15,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    full = ""
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if stream:
                print("\nRespuesta: ", end="", flush=True)
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        print(token, end="", flush=True)
                        full += token
                        if chunk.get("done"):
                            print()
                            break
            else:
                data = json.loads(resp.read())
                full = data.get("response", "")
    except urllib.error.URLError as e:
        print(f"\nError Ollama: {e}")
    return full


def format_context(results: list[dict]) -> str:
    """
    Formatea los chunks como contexto legible.
    Incluye seccion y pagina como referencia para el modelo.
    """
    if not results:
        return "No se encontraron fragmentos relevantes."

    parts = []
    total = 0

    for r in results:
        if total >= MAX_CONTEXT_CHARS:
            break

        header = f"[Seccion: {r['section']} | Pagina {r['page']}]"

        if r["type"] == "table":
            body = f"TABLA — Columnas: {' | '.join(r.get('headers', []))}\n"
            for row in r.get("rows", [])[:10]:
                row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
                if row_str:
                    body += row_str + "\n"
        else:
            body = r["content"]

        block = f"{header}\n{body}"
        remaining = MAX_CONTEXT_CHARS - total
        parts.append(block[:remaining])
        total += len(block)

    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}

[CONTEXTO DEL MANUAL]
{context}
[FIN DEL CONTEXTO]

Pregunta: {question}

Respuesta:"""


def run_chat(index: TFIDFIndex, model: str, debug: bool) -> None:
    print("\n" + "="*60)
    print("  ASISTENTE TECNICO — MANUAL HIOKI CM4001")
    print("="*60)
    print("  Comandos: 'fuentes' | 'contexto' | 'salir'")
    print("="*60)

    last_sources = []
    last_context = ""

    while True:
        try:
            print()
            q = input("Tu pregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego.")
            break

        if not q:
            continue
        if q.lower() in ("salir", "exit"):
            print("Hasta luego.")
            break
        if q.lower() == "fuentes":
            for s in last_sources:
                print(f"  [p{s['page']}] {s['section']} (score={s['score']})")
            continue
        if q.lower() == "contexto":
            print("\n" + "-"*50)
            print(last_context)
            print("-"*50)
            continue

        # Buscar con expansion de query
        results = index.search(q, top_k=TOP_K_CONTEXT)
        last_sources = results

        context = format_context(results)
        last_context = context

        if debug:
            print(f"\n[DEBUG] Tokens query: {__import__('index_utils').tokenize(q)}")
            print(f"[DEBUG] Paginas: {[r['page'] for r in results]}")

        if results:
            pages = list(dict.fromkeys(f"p{r['page']}" for r in results))
            print(f"Contexto: {' | '.join(pages)}")

        prompt = build_prompt(q, context)
        query_ollama(prompt, model, stream=True)


# ── Extraccion estructurada ────────────────────────────────────────────────────

FIELDS = [
    {
        "field":  "tipo_instrumento",
        "query":  "pinza amperimetrica fugas CA descripcion producto tipo instrumento",
        "task":   'Extrae el tipo exacto del instrumento. JSON: {"tipo": "...", "descripcion_corta": "..."}'
    },
    {
        "field":  "rango_corriente",
        "query":  "rango medicion corriente minimo maximo mA amperios",
        "task":   'Extrae rango de medicion. JSON: {"minimo": "...", "maximo": "...", "unidad": "..."}'
    },
    {
        "field":  "categoria_medicion",
        "query":  "categoria medicion CAT voltaje clasificacion seguridad",
        "task":   'Extrae categoria CAT y voltaje. JSON: {"categoria": "...", "voltaje_v": "..."}'
    },
    {
        "field":  "diametro_mordaza",
        "query":  "diametro maximo mordaza abrazadera milimetros cable",
        "task":   'Extrae diametro maximo. JSON: {"diametro_max_mm": "..."}'
    },
    {
        "field":  "rangos_secuencia",
        "query":  "rangos AUTO mA amperios secuencia RANGE tecla",
        "task":   'Lista los rangos en orden. JSON: {"rangos": ["60,00 mA", "600,0 mA", ...]}'
    },
    {
        "field":  "bluetooth",
        "query":  "comunicacion inalambrica bluetooth Z3210 GENNECT metros frecuencia",
        "task":   'Extrae datos bluetooth. JSON: {"adaptador": "...", "app": "...", "frecuencia_ghz": "...", "rango_m": "..."}'
    },
    {
        "field":  "funciones",
        "query":  "funciones principales caracteristicas capacidades instrumento medicion",
        "task":   'Lista funciones principales. JSON: {"funciones": ["...", "..."]}'
    },
]


def extract_json(text: str) -> dict:
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    m = re.search(r'\[[^\[\]]+\]', text, re.DOTALL)
    if m:
        try:
            return {"lista": json.loads(m.group())}
        except Exception:
            pass
    return {"raw": text.strip()[:300]}


def run_extraction(index: TFIDFIndex, model: str) -> None:
    print("\n" + "="*60)
    print("  EXTRACCION DE DATOS ESTRUCTURADOS")
    print("="*60)

    data = {}
    for fd in FIELDS:
        print(f"  Extrayendo: {fd['field']}...", end=" ", flush=True)

        results = index.search(fd["query"], top_k=5)
        context = format_context(results)

        prompt = f"""{SYSTEM_PROMPT}

[CONTEXTO DEL MANUAL]
{context}
[FIN DEL CONTEXTO]

Tarea: {fd['task']}
Responde UNICAMENTE con el JSON pedido. Sin texto adicional. Sin markdown.
Si el dato no esta en el contexto escribe null.

JSON:"""

        resp = query_ollama(prompt, model, stream=False)
        data[fd["field"]] = extract_json(resp)
        print("OK")

    out = Path("output/extracted_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nGuardado: {out}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not INDEX_FILE.exists():
        print("Indice no encontrado. Ejecuta:")
        print("  python 1_extract_pdf.py manuales\\CM4001A966-01.pdf")
        print("  python 2_build_index.py")
        sys.exit(1)

    print("Cargando indice...")
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)
    print(f"  {len(index.documents)} documentos")

    print(f"Verificando Ollama ({args.model})...")
    if not check_ollama(args.model):
        sys.exit(1)
    print(f"  OK")

    if args.extract:
        run_extraction(index, args.model)
    else:
        run_chat(index, args.model, args.debug)


if __name__ == "__main__":
    main()