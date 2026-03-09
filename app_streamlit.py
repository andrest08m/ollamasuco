"""
app_streamlit.py — SucoBot | Asistente HIOKI Multi-Producto
============================================================
Asistente técnico-comercial con personalidad fija.
Responde saludos, preguntas técnicas y comerciales en un solo modo.

Uso:
    streamlit run app_streamlit.py
"""

import json
import pickle
import re
import urllib.request
import urllib.error
from pathlib import Path

import streamlit as st


# ── Config ────────────────────────────────────────────────────────────────────
INDEX_FILE     = Path("output/index.pkl")
PRODUCTS_FILE  = Path("output/products.json")
OLLAMA_URL     = "http://localhost:11434/api/generate"
DEFAULT_MODEL  = "qwen2.5:14b"   # ~9GB RAM cuantizado — ideal para servidor CPU 128GB
MAX_CTX_CHARS  = 3000
TIMEOUT_STREAM = 300


# ── Detección de intención casual ────────────────────────────────────────────

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
            "¡Hola! Soy **SucoBot** 🤖, el asistente técnico-comercial de **Suco** "
            "especializado en instrumentos de medición **HIOKI**.\n\n"
            "Puedo ayudarte con:\n"
            "- 🔧 **Especificaciones técnicas** — rangos, categorías CAT, funciones, dimensiones\n"
            "- 💼 **Asesoría comercial** — qué equipo recomendar, ventajas, perfil de cliente\n"
            "- 🔍 **Diagnóstico** — qué instrumento usar según el problema que describes\n\n"
            "¿Sobre qué equipo HIOKI quieres saber?"
        )
    if re.search(r'gracias?|thank', t):
        return "¡Con gusto! Si tienes más preguntas sobre los equipos HIOKI, aquí estoy. 🤝"
    if re.search(r'adios?|chao|hasta', t):
        return "¡Hasta luego! Cualquier consulta sobre HIOKI, vuelve cuando quieras. 👋"
    if re.search(r'ayuda|help', t):
        return (
            "Claro, estoy aquí para ayudarte. Puedes preguntarme cosas como:\n\n"
            "- *¿Cuál es el rango de medición de la CM4001?*\n"
            "- *¿Qué equipo uso para medir corriente de fuga en un tablero CAT III?*\n"
            "- *¿Cuáles son las ventajas de la pinza amperimétrica HIOKI?*\n\n"
            "¿Por dónde empezamos?"
        )
    # Saludo genérico
    return (
        "¡Hola! Soy **SucoBot** ⚡, tu asistente HIOKI.\n\n"
        "Estoy listo para responder tus preguntas técnicas o comerciales "
        "sobre instrumentos de medición HIOKI. ¿En qué te puedo ayudar?"
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
Transforma las especificaciones en beneficios prácticos para el cliente.
Ejemplo: "Rango AUTO hasta 600A → apto para entornos industriales sin reconfigurar el equipo"
Da la lista directamente, sin frases introductorias.

REGLA 6 — FUENTE AL FINAL
Termina siempre con: "Fuente: [PRODUCTO | p.XX]"
Si respondiste "No tengo información...", NO pongas fuente.

REGLA 7 — CONCISIÓN
Ve directo al grano. Sin introducciones. Usa viñetas cuando aplique.
Responde en el idioma de la pregunta."""


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_index():
    if not INDEX_FILE.exists():
        return None
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_products():
    if not PRODUCTS_FILE.exists():
        return {}
    with open(PRODUCTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_ollama_models():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def format_context(results: list[dict], max_chars: int = MAX_CTX_CHARS) -> str:
    if not results:
        return "No se encontraron fragmentos relevantes."
    parts = []
    total = 0
    for r in results:
        if total >= max_chars:
            break
        label = f"[{r.get('product','?')} | {r['section']} | p.{r['page']}]"
        if r["type"] == "table":
            body = f"TABLA — {' | '.join(r.get('headers', []))}\n"
            for row in r.get("rows", [])[:8]:
                row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
                if row_str:
                    body += row_str + "\n"
        else:
            body = r["content"]
        block = f"{label}\n{body}"
        remaining = max_chars - total
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


def query_ollama_stream(prompt: str, model: str):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature":    0.1,
            "num_predict":    512,
            "num_ctx":        3072,
            "repeat_penalty": 1.1,
        }
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_STREAM) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
    except Exception as e:
        yield f"\n[Error de conexión con Ollama: {e}]"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SucoBot — Asistente HIOKI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
section[data-testid="stSidebar"] { background: #0a0e1a; border-right: 1px solid #1a2235; }
.hioki-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #e11d2222;
    border-left: 4px solid #e11d22;
    border-radius: 4px 8px 8px 4px;
    padding: 16px 20px;
    margin-bottom: 20px;
}
.hioki-header h1 {
    color: #f8fafc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    margin: 0;
    letter-spacing: 0.1em;
}
.hioki-header p { color: #64748b; font-size: 0.78rem; margin: 4px 0 0 0; }
.msg-user {
    background: #0f172a;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    color: #e2e8f0;
    font-size: 0.9rem;
}
.msg-bot {
    background: #0d1320;
    border-left: 3px solid #e11d22;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin: 10px 0;
    color: #e2e8f0;
    font-size: 0.9rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.msg-casual {
    background: #0d1320;
    border-left: 3px solid #8b5cf6;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin: 10px 0;
    color: #e2e8f0;
    font-size: 0.9rem;
    line-height: 1.7;
}
.bot-badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
    background: #e11d2222;
    color: #e11d22;
    border: 1px solid #e11d2244;
}
.casual-badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
    background: #8b5cf622;
    color: #a78bfa;
    border: 1px solid #8b5cf644;
}
.source-pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
}
.source-pill.table { border-color: #f59e0b44; color: #fbbf24; }
.product-tag {
    display: inline-block;
    background: #1d4ed822;
    border: 1px solid #3b82f644;
    color: #93c5fd;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 4px;
    margin: 2px;
}
.stat-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}
.stat-box .val { font-size: 1.6rem; font-weight: 600; color: #60a5fa; font-family: 'IBM Plex Mono', monospace; }
.stat-box .lbl { font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "product_filter" not in st.session_state:
    st.session_state.product_filter = None


# ── Load ──────────────────────────────────────────────────────────────────────
index     = load_index()
products  = load_products()
models    = get_ollama_models()
ollama_ok = len(models) > 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="hioki-header">
      <h1>⚡ SucoBot</h1>
      <p>Asistente Técnico-Comercial HIOKI</p>
    </div>
    """, unsafe_allow_html=True)

    if index:
        n_docs   = len(index.documents)
        n_prods  = len(index.products) if hasattr(index, 'products') else len(products)
        n_tables = sum(1 for d in index.documents if d.get("type") == "table")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="val">{n_prods}</div><div class="lbl">Equipos</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="val">{n_docs}</div><div class="lbl">Chunks</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="val">{n_tables}</div><div class="lbl">Tablas</div></div>', unsafe_allow_html=True)
    else:
        st.error("Índice no encontrado")
        st.code("python 1_extract_pdf.py manuales/\npython 2_build_index.py")

    st.divider()

    st.markdown("**Filtrar por producto**")
    prod_options = ["Todos los productos"] + (
        index.products if index and hasattr(index, 'products') else list(products.keys())
    )
    selected_prod = st.selectbox("Producto", prod_options, label_visibility="collapsed")
    st.session_state.product_filter = None if selected_prod == "Todos los productos" else selected_prod

    st.divider()

    st.markdown("**Modelo**")
    if ollama_ok:
        st.markdown('<span style="color:#10b981;font-size:0.8rem">● Ollama activo</span>', unsafe_allow_html=True)
        default_idx = next(
            (i for i, m in enumerate(models) if "qwen2.5:14b" in m),
            next((i for i, m in enumerate(models) if "qwen" in m),
            next((i for i, m in enumerate(models) if "llama3.2" in m), 0))
        )
        model_name = st.selectbox("Modelo", models, label_visibility="collapsed", index=default_idx)
    else:
        st.markdown('<span style="color:#ef4444;font-size:0.8rem">● Ollama no disponible</span>', unsafe_allow_html=True)
        model_name = st.text_input("Modelo", value=DEFAULT_MODEL, label_visibility="collapsed")

    cpu_mode = st.toggle("Modo CPU (contexto reducido)", value=True,
                          help="Reduce contexto para CPUs sin GPU — recomendado en servidor CPU")

    st.divider()

    if st.session_state.last_sources:
        st.markdown("**Fuentes usadas**")
        for r in st.session_state.last_sources:
            tipo_class = "table" if r.get("type") == "table" else ""
            tipo_icon  = "📊" if r.get("type") == "table" else "📄"
            prod = r.get("product", "?")
            st.markdown(
                f'<span class="source-pill {tipo_class}">{tipo_icon} {prod} p.{r["page"]}'
                f' ({r.get("score",0):.2f})</span>',
                unsafe_allow_html=True
            )

    st.divider()

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid #1e293b;margin-bottom:16px">
  <span style="font-size:1.8rem">⚡</span>
  <div>
    <div style="color:#f8fafc;font-size:1.1rem;font-weight:600">SucoBot — Asistente HIOKI</div>
    <div style="color:#64748b;font-size:0.78rem">Preguntas técnicas, comerciales y de diagnóstico sobre instrumentos HIOKI</div>
  </div>
  <span style="margin-left:auto;background:#1e293b;border:1px solid #334155;color:#94a3b8;font-size:0.7rem;padding:4px 10px;border-radius:20px;font-family:IBM Plex Mono">Multi-producto</span>
</div>
""", unsafe_allow_html=True)


# ── Historial ─────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        if msg.get("casual"):
            st.markdown(
                f'<div class="msg-casual"><span class="casual-badge">SucoBot</span><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            prod_tags = "".join(f'<span class="product-tag">{p}</span>' for p in msg.get("products", []))
            st.markdown(
                f'<div class="msg-bot"><span class="bot-badge">SucoBot ⚡</span> {prod_tags}<br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )


# ── Input ─────────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col_q, col_btn = st.columns([6, 1])
    with col_q:
        user_input = st.text_input(
            "Pregunta",
            placeholder="Ej: ¿Cuál es el rango de medición? ¿A qué cliente le recomiendo este equipo? o simplemente saluda",
            label_visibility="collapsed"
        )
    with col_btn:
        submitted = st.form_submit_button("→", type="primary", use_container_width=True)

QUICK_QUESTIONS = [
    "¿Qué tipo de instrumento es la CM4001?",
    "¿Cuáles son sus principales ventajas?",
    "¿Qué categoría CAT tiene y qué significa?",
    "Tengo disparos intermitentes en un diferencial, ¿qué equipo usar?",
]
st.markdown("**Preguntas rápidas:**")
qcols = st.columns(2)
for i, q in enumerate(QUICK_QUESTIONS):
    with qcols[i % 2]:
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            user_input = q
            submitted = True


# ── Procesar ──────────────────────────────────────────────────────────────────
if submitted and user_input and user_input.strip():

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Caso 1: Mensaje casual — sin índice, respuesta directa
    if is_casual(user_input):
        response = casual_response(user_input)
        st.session_state.last_sources = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "casual": True,
            "products": [],
        })
        st.rerun()

    # Caso 2: Pregunta técnica/comercial — RAG
    elif index:
        pf = st.session_state.product_filter

        stats = index.search_with_stats(
            user_input,
            top_k=6,
            product_filter=pf,
            multi_product=(pf is None)
        )
        results = stats["results"]
        st.session_state.last_sources = results

        max_chars = 1600 if cpu_mode else MAX_CTX_CHARS
        context = format_context(results, max_chars)

        prods_found = stats.get("products_found", [])
        if prods_found:
            prod_html = " ".join(f'<span class="product-tag">{p}</span>' for p in prods_found)
            pages = list(dict.fromkeys(f"p{r['page']}" for r in results))
            st.markdown(
                f'<div style="margin:4px 0 8px;font-size:0.75rem;color:#64748b">'
                f'Contexto: {prod_html} — {" · ".join(pages)}</div>',
                unsafe_allow_html=True
            )

        prompt = build_prompt(user_input, context)

        with st.spinner("SucoBot procesando..."):
            full_response = ""
            placeholder = st.empty()

            if ollama_ok:
                for token in query_ollama_stream(prompt, model_name):
                    full_response += token
                    placeholder.markdown(
                        f'<div class="msg-bot"><span class="bot-badge">SucoBot ⚡</span><br>{full_response}▌</div>',
                        unsafe_allow_html=True
                    )
                placeholder.markdown(
                    f'<div class="msg-bot"><span class="bot-badge">SucoBot ⚡</span><br>{full_response}</div>',
                    unsafe_allow_html=True
                )
            else:
                full_response = f"⚠️ Ollama no disponible.\n\nContexto recuperado:\n\n{context[:800]}..."
                placeholder.markdown(f'<div class="msg-bot">{full_response}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "casual": False,
            "products": prods_found,
        })
        st.rerun()

    else:
        st.error("Índice no cargado.\n\n`python 1_extract_pdf.py manuales/`\n\n`python 2_build_index.py`")