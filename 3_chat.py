"""
3_chat.py — SucoBot CLI | RAG anti-alucinacion con Ollama
=========================================================
Asistente técnico-comercial HIOKI en modo terminal.
Detecta automáticamente saludos vs preguntas técnicas/comerciales.

Uso:
    python 3_chat.py
    python 3_chat.py --model qwen2.5:14b
    python 3_chat.py --debug
    python 3_chat.py --extract
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
DEFAULT_MODEL = "qwen2.5:14b"        # ~9GB RAM cuantizado — ideal para 128GB CPU
INDEX_FILE    = Path("output/index.pkl")
TOP_K_CONTEXT = 6
MAX_CONTEXT_CHARS = 3500


# ── Detección de intención casual ─────────────────────────────────────────────

CASUAL_PATTERNS = re.compile(
    r'^\s*(hola|hey|buenas?|buenos?\s*(días?|tardes?|noches?)|'
    r'que\s*tal|como\s*est(as?|á)|buen\s*día|saludos?|hi\b|hello\b|'
    r'gracias?|thank|adios?|chao|hasta\s*(luego|pronto|mañana)|'
    r'quien\s*(eres?|es\s*usted)|como\s*te\s*llamas?|tu\s*nombre|'
    r'que\s*(eres?|haces?|puedes?)|para\s*que\s*sirves?|'
    r'ayuda|help\b|que\s*sabes?)\s*[?!.]*\s*$',
    re.IGNORECASE | re.UNICODE
)

def is_casual(text: str) -> bool:
    return bool(CASUAL_PATTERNS.match(text.strip()))


def casual_response(text: str) -> str:
    t = text.lower().strip()
    if re.search(r'quien|llamas?|nombre|eres?|que\s*haces?|para\s*que|que\s*sabes?|puedes?', t):
        return (
            "¡Hola! Soy SucoBot 🤖, el asistente técnico-comercial de Suco\n"
            "especializado en instrumentos de medición HIOKI.\n\n"
            "Puedo ayudarte con:\n"
            "  🔧 Especificaciones técnicas — rangos, CAT, funciones, dimensiones\n"
            "  💼 Asesoría comercial       — qué equipo recomendar, ventajas\n"
            "  🔍 Diagnóstico              — qué instrumento usar según el problema\n\n"
            "¿Sobre qué equipo HIOKI quieres saber?"
        )
    if re.search(r'gracias?|thank', t):
        return "¡Con gusto! Si tienes más preguntas sobre HIOKI, aquí estoy. 🤝"
    if re.search(r'adios?|chao|hasta', t):
        return "¡Hasta luego! Vuelve cuando tengas consultas sobre HIOKI. 👋"
    if re.search(r'ayuda|help', t):
        return (
            "Puedes preguntarme cosas como:\n"
            "  • ¿Cuál es el rango de medición de la CM4001?\n"
            "  • ¿Qué equipo uso para un tablero CAT III?\n"
            "  • ¿Cuáles son las ventajas de la pinza amperimétrica HIOKI?\n\n"
            "¿Por dónde empezamos?"
        )
    return (
        "¡Hola! Soy SucoBot ⚡, tu asistente HIOKI.\n"
        "Listo para responder preguntas técnicas o comerciales.\n"
        "¿En qué te puedo ayudar?"
    )


# ── System Prompt SucoBot ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres SucoBot, el asistente técnico-comercial de Suco especializado en instrumentos de medición HIOKI.
Tu personalidad: profesional, directo, confiable. Respondes tanto preguntas técnicas como comerciales.

REGLAS CRÍTICAS — DE CUMPLIMIENTO OBLIGATORIO:

REGLA 1 — SOLO USA EL CONTEXTO
Responde ÚNICAMENTE con información del bloque [CONTEXTO DEL MANUAL].
No uses conocimiento externo. No inventes datos técnicos, rangos ni especificaciones.

REGLA 2 — SI ESTÁ EN EL CONTEXTO, RESPÓNDELO
Lee TODO el contexto antes de responder. Si la información está aunque sea parcialmente, úsala.

REGLA 3 — SI NO ESTÁ EN EL CONTEXTO
Di exactamente: "No tengo esa información en los manuales disponibles."
Nunca uses "probablemente", "generalmente" ni completes con suposiciones.

REGLA 4 — DATOS TÉCNICOS: VALORES EXACTOS
Copia los valores exactos del contexto: rangos, voltajes, dimensiones, categorías.
Ejemplo correcto: "mide desde 0,60 mA hasta 600,0 A"
Ejemplo incorrecto: "mide corrientes pequeñas y grandes"

REGLA 5 — VENTAJAS Y RECOMENDACIONES COMERCIALES
Cuando te pregunten ventajas, beneficios o a quién recomendar:
Transforma especificaciones en beneficios prácticos para el cliente.
Ejemplo: "Rango AUTO hasta 600A → apto para entornos industriales sin reconfigurar el equipo"
Da la lista directamente, sin frases introductorias.

REGLA 6 — FUENTE AL FINAL
Termina siempre con: "Fuente: [PRODUCTO | p.XX]"
Si respondiste "No tengo información...", NO pongas fuente.

REGLA 7 — CONCISIÓN
Ve directo al grano. Sin introducciones. Usa viñetas cuando aplique.
Responde en el idioma de la pregunta."""


# ── Ollama ────────────────────────────────────────────────────────────────────

def check_ollama(model: str) -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if not any(model in m for m in models):
                print(f"Modelo '{model}' no encontrado.")
                print(f"  Disponibles: {models}")
                print(f"  Instala con: ollama pull {model}")
                return False
            return True
    except Exception as e:
        print(f"Ollama no responde: {e}")
        print("  Inicia con: ollama serve")
        return False


def query_ollama(prompt: str, model: str, stream: bool = True) -> str:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature":    0.1,
            "num_predict":    512,
            "num_ctx":        3072,
            "repeat_penalty": 1.1,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    full = ""
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if stream:
                print("\nSucoBot: ", end="", flush=True)
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


# ── RAG ───────────────────────────────────────────────────────────────────────

def format_context(results: list[dict]) -> str:
    if not results:
        return "No se encontraron fragmentos relevantes."
    parts = []
    total = 0
    for r in results:
        if total >= MAX_CONTEXT_CHARS:
            break
        label = f"[{r.get('product','?')} | {r['section']} | p.{r['page']}]"
        if r["type"] == "table":
            body = f"TABLA — {' | '.join(r.get('headers', []))}\n"
            for row in r.get("rows", [])[:10]:
                row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
                if row_str:
                    body += row_str + "\n"
        else:
            body = r["content"]
        block = f"{label}\n{body}"
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


# ── Chat interactivo ──────────────────────────────────────────────────────────

def run_chat(index: TFIDFIndex, model: str, debug: bool) -> None:
    print("\n" + "="*62)
    print("  SucoBot ⚡ — Asistente Técnico-Comercial HIOKI")
    print("="*62)
    print("  Pregunta lo que quieras: técnico, comercial o saluda.")
    print("  Comandos: 'fuentes' | 'contexto' | 'salir'")
    print("="*62)

    last_sources = []
    last_context = ""

    while True:
        try:
            print()
            q = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSucoBot: ¡Hasta luego! 👋")
            break

        if not q:
            continue
        if q.lower() in ("salir", "exit", "quit"):
            print("SucoBot: ¡Hasta luego! 👋")
            break
        if q.lower() == "fuentes":
            if last_sources:
                for s in last_sources:
                    tipo = "TABLA" if s.get("type") == "table" else "TEXTO"
                    print(f"  [{tipo}] {s.get('product','?')} p.{s['page']} | {s['section']} (score={s['score']})")
            else:
                print("  Sin fuentes disponibles.")
            continue
        if q.lower() == "contexto":
            if last_context:
                print("-" * 50)
                print(last_context[:1000])
                print("-" * 50)
            else:
                print("  Sin contexto disponible.")
            continue

        # Detectar casual
        if is_casual(q):
            print(f"\nSucoBot: {casual_response(q)}")
            last_sources = []
            last_context = ""
            continue

        # RAG técnico/comercial
        results = index.search(q, top_k=TOP_K_CONTEXT)
        last_sources = results

        context = format_context(results)
        last_context = context

        if debug:
            from index_utils import tokenize
            print(f"\n[DEBUG] Tokens: {tokenize(q)}")
            print(f"[DEBUG] Páginas: {[(r.get('product'), r['page']) for r in results]}")

        if results:
            prods = list(dict.fromkeys(r.get("product", "?") for r in results))
            pages = list(dict.fromkeys(f"p{r['page']}" for r in results))
            print(f"Contexto: {' | '.join(prods)} — {' · '.join(pages)}")

        prompt = build_prompt(q, context)
        query_ollama(prompt, model, stream=True)


# ── Extracción estructurada ───────────────────────────────────────────────────

FIELDS = [
    {"field": "tipo_instrumento",
     "query": "pinza amperimetrica fugas CA descripcion producto tipo instrumento",
     "task":  'JSON: {"tipo": "...", "descripcion_corta": "..."}'},
    {"field": "rango_corriente",
     "query": "rango medicion corriente minimo maximo mA amperios",
     "task":  'JSON: {"minimo": "...", "maximo": "...", "unidad": "..."}'},
    {"field": "categoria_medicion",
     "query": "categoria medicion CAT voltaje clasificacion seguridad",
     "task":  'JSON: {"categoria": "...", "voltaje_v": "..."}'},
    {"field": "diametro_mordaza",
     "query": "diametro maximo mordaza abrazadera milimetros cable",
     "task":  'JSON: {"diametro_max_mm": "..."}'},
    {"field": "rangos_secuencia",
     "query": "rangos AUTO mA amperios secuencia RANGE tecla",
     "task":  'JSON: {"rangos": ["60,00 mA", "600,0 mA", ...]}'},
    {"field": "bluetooth",
     "query": "comunicacion inalambrica bluetooth Z3210 GENNECT metros frecuencia",
     "task":  'JSON: {"adaptador": "...", "app": "...", "frecuencia_ghz": "...", "rango_m": "..."}'},
    {"field": "funciones",
     "query": "funciones principales caracteristicas capacidades instrumento medicion",
     "task":  'JSON: {"funciones": ["...", "..."]}'},
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
    print("\n" + "="*62)
    print("  SucoBot — EXTRACCIÓN DE DATOS ESTRUCTURADOS")
    print("="*62)
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
Responde ÚNICAMENTE con el JSON pedido. Sin texto adicional. Sin markdown.
Si el dato no está en el contexto escribe null.

JSON:"""
        resp = query_ollama(prompt, model, stream=False)
        data[fd["field"]] = extract_json(resp)
        print("OK")

    out = Path("output/extracted_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {out}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SucoBot — Asistente HIOKI CLI")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    if not INDEX_FILE.exists():
        print("Índice no encontrado. Ejecuta:")
        print("  python 1_extract_pdf.py manuales/")
        print("  python 2_build_index.py")
        sys.exit(1)

    print("Cargando índice...")
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)
    prods = ', '.join(getattr(index, 'products', []))
    print(f"  {len(index.documents)} documentos | productos: {prods}")

    print(f"Verificando Ollama ({args.model})...")
    if not check_ollama(args.model):
        sys.exit(1)
    print("  OK")

    if args.extract:
        run_extraction(index, args.model)
    else:
        run_chat(index, args.model, args.debug)


if __name__ == "__main__":
    main()