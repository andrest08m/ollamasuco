"""
3_chat.py — SucoBot CLI | RAG Agentic con Ollama
=================================================
Asistente técnico-comercial HIOKI en modo terminal.
Implementa un bucle agéntico interactivo para extracción estructurada.

Uso:
    python 3_chat.py
    python 3_chat.py --model qwen2.5:7b-instruct
    python 3_chat.py --debug
    python 3_chat.py --extract
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

from index_utils import SemanticIndex

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
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
            "num_ctx":        4096,
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
        with urllib.request.urlopen(req, timeout=180) as resp:
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

def run_chat(index: SemanticIndex, model: str, debug: bool) -> None:
    print("\n" + "="*62)
    print("  SucoBot ⚡ — Asistente HIOKI (Búsqueda Semántica v6)")
    print("="*62)
    print("  Pregunta lo que quieras: técnico, comercial o saluda.")
    print(f"  Modelo Activo: {model}")
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
                    # SemanticIndex provee 'score' del Reranker FlashRank
                    print(f"  [{tipo}] {s.get('product','?')} p.{s['page']} | {s['section']} (FlashRank score={s.get('score', 0):.4f})")
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
            print(f"\n[DEBUG] Páginas encontradas: {[(r.get('product'), r['page']) for r in results]}")

        if results:
            prods = list(dict.fromkeys(r.get("product", "?") for r in results))
            pages = list(dict.fromkeys(f"p{r['page']}" for r in results))
            print(f"Contexto recuperado: {' | '.join(prods)} — {' · '.join(pages)}")

        prompt = build_prompt(q, context)
        query_ollama(prompt, model, stream=True)


# ── Extracción Estructurada Agéntica (Bucle ReAct) ────────────────────────────────

FIELDS = [
    {"field": "tipo_instrumento", "task": 'Descripción del producto o tipo de instrumento de medición.'},
    {"field": "rango_corriente",  "task": 'Rango de medición de corriente ACA (mínimo, máximo, unidad, ejemplo: mA, A).'},
    {"field": "categoria_medicion", "task": 'Clasificación de seguridad: Categoría de medición CAT (ej. CAT III, CAT IV) y voltaje admitido o listado.'},
    {"field": "diametro_mordaza", "task": 'Diámetro máximo de la mordaza o el ancho máximo de la barra sensora en milímetros.'},
    {"field": "comunicacion",     "task": 'Soporte para Bluetooth, módulo o adaptador Z3210 y uso con app Gennect Cross.'},
    {"field": "funciones_extra",  "task": 'Lista de funciones extras (ej. retención de datos HOLD, filtro de paso bajo, comparador sonoro o zumbador).'},
]

AGENT_PROMPT = """Eres el Agente SucoBot de EXTRACCIÓN de datos HIOKI.
Tu objetivo es recolectar un dato específico de un manual leyendo contexto.
Debes operar en un bucle mental iterativo usando PENSAMIENTO y ACCION.

Tienes las siguientes herramientas a tu disposición:
1. SEARCH("texto_a_buscar"): Te permite consultar de nuevo el manual si la información que tienes no es suficiente o no aplica al dato que necesitas.
2. DONE(json_resultado): Usas esta función cuando estás completamente seguro de que en el contexto actual TIEENS el dato pedido. Debes pasar el JSON como resultado.

REGLAS PARA EL JSON:
Solo devuelve el valor crudo en código JSON, sin introducciones. Si necesitas buscar datos usa JSON para describir el SEARCH, así:
{ "action": "SEARCH", "query": "rango de medicion amperaje maximo" } o 
{ "action": "DONE", "result": { "valor_encontrado": "20V" } } o
{ "action": "DONE", "result": null } si tras 3 búsquedas no encuentras NADA.

[CONTEXTO RECOPIALDO HASTA AHORA]
{context}

[TAREA ACTUAL]
Extraer este dato en formato JSON: {task}
"""

def extract_json(text: str) -> dict:
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}

def run_agentic_extraction(index: SemanticIndex, model: str) -> None:
    print("\n" + "="*62)
    print("  SucoBot — AGENTE DE EXTRACCIÓN AUTÓNOMA (Modo ReAct)")
    print("="*62)
    data = {}
    
    max_iterations = 3
    
    for fd in FIELDS:
        print(f"\n▶ Extrayendo campo: {fd['field']}...")
        
        # Búsqueda inicial por nombre del campo
        accumulated_results = index.search(fd["task"], top_k=3)
        context = format_context(accumulated_results)
        
        iteration = 0
        final_result = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"  Iteración {iteration} [Pensando]...", end=" ", flush=True)
            
            prompt = AGENT_PROMPT.replace("{context}", context).replace("{task}", fd["task"])
            
            resp = query_ollama(prompt, model, stream=False)
            response_json = extract_json(resp)
            
            action = response_json.get("action", "DONE")
            
            if action == "SEARCH":
                query = response_json.get("query", fd["task"])
                print(f"[Buscando nuevo contexto: '{query}']")
                new_results = index.search(query, top_k=3)
                
                # Agregar nuevo contexto manteniendo el tamaño bajo control
                accumulated_results.extend(new_results)
                # Deduplicar contexto si es necesario (el Semantic Index format_context trunca el string igual)
                context = format_context(accumulated_results)
            elif action == "DONE":
                final_result = response_json.get("result")
                print(f"[Listo: {final_result}]")
                break
            else:
                # Fallback por si la respuesta no es totalmente estructurada pero tiene info
                print(f"[Fallback para respuesta cruda]")
                final_result = {"raw": resp[:200]}
                break
                
        if not final_result:
            print("  [Agent Timeout: Dato no encontrado]")
            
        data[fd["field"]] = final_result

    out = Path("output/extracted_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n============================================================")
    print(f"Extracción finalizada. Guardado: {out}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SucoBot — Asistente HIOKI CLI v6")
    parser.add_argument("--extract", action="store_true", help="Inicia extracción autónoma agéntica")
    parser.add_argument("--model",   default=DEFAULT_MODEL, help="Selecciona el modelo Ollama")
    parser.add_argument("--debug",   action="store_true", help="Activa modo debug (tiempos y páginas)")
    args = parser.parse_args()

    if not INDEX_FILE.exists():
        print("Índice no encontrado. Ejecuta:")
        print("  python 1_extract_pdf.py manuales/")
        print("  python 2_build_index.py")
        sys.exit(1)

    print("Cargando índice semántico...")
    try:
        index = SemanticIndex.load(INDEX_FILE)
    except Exception as e:
        print(f"Error cargando índice: {e}")
        print("Debes ejecutar 'python 2_build_index.py' primero para generar la base factorial.")
        sys.exit(1)
        
    prods = ', '.join(getattr(index, 'products', []))
    print(f"  {len(index.documents)} documentos vectoriales | productos: {prods}")

    print(f"Verificando Ollama ({args.model})...")
    if not check_ollama(args.model):
        sys.exit(1)
    print("  OK")

    if args.extract:
        run_agentic_extraction(index, args.model)
    else:
        run_chat(index, args.model, args.debug)


if __name__ == "__main__":
    main()