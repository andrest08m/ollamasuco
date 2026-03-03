"""
app_streamlit.py v5 — Asistente HIOKI Multi-Producto
=====================================================
Un solo chat con 3 modos:
  🔧 Técnico   — especificaciones exactas, paso a paso
  💼 Ventas    — recomendaciones, perfil de cliente, ventajas
  🔍 Diagnóstico — "tengo este problema → ¿qué equipo uso?"

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
INDEX_FILE    = Path("output/index.pkl")
PRODUCTS_FILE = Path("output/products.json")
OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"
MAX_CTX_CHARS = 2800
TIMEOUT_STREAM = 300
TIMEOUT_SYNC   = 180

# ── System Prompts por modo ───────────────────────────────────────────────────

PROMPTS = {
    "tecnico": """Eres un ingeniero técnico experto en instrumentos de medición HIOKI.
Tu fuente es el [CONTEXTO DEL MANUAL] a continuación.

REGLAS:
1. Responde con datos EXACTOS del contexto: valores numéricos, unidades, procedimientos paso a paso.
2. Si la pregunta es sobre especificaciones, cita el valor exacto (ej: "60,00 mA", "CAT III 300V").
3. Si hay un procedimiento, lista los pasos en orden.
4. Si el dato no está en el contexto: dilo claramente y sugiere dónde podría encontrarse.
5. NO empieces con un número suelto. Responde en el idioma de la pregunta.""",

    "ventas": """Eres un asesor comercial experto en instrumentos de medición HIOKI.
Tu fuente es el [CONTEXTO DEL MANUAL] a continuación.

REGLAS:
1. Responde pensando en el cliente: beneficios, casos de uso, perfil del comprador ideal.
2. Para "¿a quién se recomienda?": identifica el tipo de profesional o industria según las aplicaciones del instrumento.
3. Para "¿qué ventajas tiene?": destaca las características diferenciadoras del contexto.
4. Para "¿cuál es mejor?": compara basándote en los datos del contexto de cada producto.
5. Puedes razonar y hacer recomendaciones siempre que se basen en el contexto.
6. Si el contexto no tiene suficiente info para comparar, dilo y describe lo que sí sabes.
7. Responde en el idioma de la pregunta.""",

    "diagnostico": """Eres un experto en diagnóstico y selección de instrumentos HIOKI.
Tu fuente es el [CONTEXTO DEL MANUAL] a continuación.

REGLAS:
1. El usuario describió un problema o necesidad. Tu trabajo es recomendar el equipo o función correcta.
2. Analiza el contexto de TODOS los productos mencionados y selecciona el más adecuado.
3. Estructura tu respuesta así:
   - PROBLEMA IDENTIFICADO: resume el problema del usuario
   - EQUIPO RECOMENDADO: nombre del producto y por qué
   - CÓMO USARLO: pasos clave del manual para ese caso
   - ALTERNATIVAS: si hay otro equipo que también podría servir
4. Si ningún equipo del contexto resuelve el problema, dilo claramente.
5. Responde en el idioma de la pregunta."""
}

MODE_CONFIG = {
    "tecnico":     {"icon": "🔧", "label": "Técnico",     "color": "#3b82f6", "multi": False},
    "ventas":      {"icon": "💼", "label": "Ventas",      "color": "#10b981", "multi": True},
    "diagnostico": {"icon": "🔍", "label": "Diagnóstico", "color": "#f59e0b", "multi": True},
}

QUICK_QUESTIONS = {
    "tecnico": [
        "¿Cuál es el rango mínimo y máximo de corriente?",
        "¿Qué categoría CAT tiene y qué significa?",
        "¿Cómo se usa la función HOLD?",
        "¿Qué tipo de baterías usa y cuántas?",
    ],
    "ventas": [
        "¿A qué tipo de cliente le recomendarías este equipo?",
        "¿Cuáles son las principales ventajas de este instrumento?",
        "¿Qué lo diferencia de otros medidores de corriente?",
        "¿Para qué industrias es ideal?",
    ],
    "diagnostico": [
        "Necesito detectar fugas en un circuito trifásico industrial",
        "Tengo disparos intermitentes en un interruptor diferencial",
        "Quiero medir corriente de arranque de un motor",
        "Necesito transferir datos de medición directamente a Excel",
    ],
}


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
        prod_label = f"[{r.get('product','?')} | {r['section']} | p.{r['page']}]"
        if r["type"] == "table":
            body = f"TABLA — {' | '.join(r.get('headers', []))}\n"
            for row in r.get("rows", [])[:8]:
                row_str = " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
                if row_str:
                    body += row_str + "\n"
        else:
            body = r["content"]
        block = f"{prod_label}\n{body}"
        remaining = max_chars - total
        parts.append(block[:remaining])
        total += len(block)
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context: str, mode: str) -> str:
    return f"""{PROMPTS[mode]}

[CONTEXTO DEL MANUAL]
{context}
[FIN DEL CONTEXTO]

Pregunta: {question}

Respuesta:"""


def query_ollama_stream(prompt: str, model: str):
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": True,
        "options": {"temperature": 0.1, "num_predict": 400, "num_ctx": 2048,
                    "repeat_penalty": 1.1}
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_STREAM) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
    except Exception as e:
        yield f"\n[Error: {e}]"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HIOKI — Asistente Técnico",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

section[data-testid="stSidebar"] {
    background: #0a0e1a;
    border-right: 1px solid #1a2235;
}

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

.mode-btn-active {
    background: #1e293b;
    border: 2px solid var(--mode-color);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 3px 0;
    cursor: pointer;
}

.msg-user {
    background: #0f172a;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    color: #e2e8f0;
    font-size: 0.9rem;
}
.msg-assistant {
    background: #0d1320;
    border-left: 3px solid var(--resp-color, #10b981);
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin: 10px 0;
    color: #e2e8f0;
    font-size: 0.9rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.msg-mode-badge {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
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
.stat-box .val {
    font-size: 1.6rem;
    font-weight: 600;
    color: #60a5fa;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-box .lbl { font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; }

.quick-q {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.8rem;
    color: #94a3b8;
    cursor: pointer;
    margin: 3px 0;
    transition: border-color 0.15s;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "tecnico"
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "product_filter" not in st.session_state:
    st.session_state.product_filter = None


# ── Load data ─────────────────────────────────────────────────────────────────
index    = load_index()
products = load_products()
models   = get_ollama_models()
ollama_ok = len(models) > 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="hioki-header">
      <h1>⚡ HIOKI</h1>
      <p>Asistente Técnico Multi-Producto v5</p>
    </div>
    """, unsafe_allow_html=True)

    # Estado
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

    # Selector de MODO
    st.markdown("**Modo de asistencia**")
    for mode_key, cfg in MODE_CONFIG.items():
        selected = st.session_state.mode == mode_key
        label = f"{cfg['icon']} {cfg['label']}"
        if selected:
            label = "✓ " + label
        if st.button(label, key=f"mode_{mode_key}",
                     type="primary" if selected else "secondary",
                     use_container_width=True):
            if st.session_state.mode != mode_key:
                st.session_state.mode = mode_key
                st.rerun()

    st.divider()

    # Filtro de producto
    st.markdown("**Filtrar por producto**")
    prod_options = ["Todos los productos"] + (index.products if index and hasattr(index, 'products') else list(products.keys()))
    selected_prod = st.selectbox("", prod_options, label_visibility="collapsed")
    st.session_state.product_filter = None if selected_prod == "Todos los productos" else selected_prod

    st.divider()

    # Modelo Ollama
    st.markdown("**Modelo**")
    if ollama_ok:
        st.markdown('<span style="color:#10b981;font-size:0.8rem">● Ollama activo</span>', unsafe_allow_html=True)
        model_name = st.selectbox("", models, label_visibility="collapsed",
                                   index=next((i for i, m in enumerate(models) if "llama3.2" in m), 0))
    else:
        st.markdown('<span style="color:#ef4444;font-size:0.8rem">● Ollama no disponible</span>', unsafe_allow_html=True)
        model_name = st.text_input("", value=DEFAULT_MODEL, label_visibility="collapsed")

    cpu_mode = st.toggle("Modo CPU", value=True, help="Reduce contexto para CPUs lentas")

    st.divider()

    # Fuentes de la última respuesta
    if st.session_state.last_sources:
        st.markdown("**Fuentes usadas**")
        max_score = max(r.get("score", 1) for r in st.session_state.last_sources) or 1
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


# ── Main chat ─────────────────────────────────────────────────────────────────
mode_cfg = MODE_CONFIG[st.session_state.mode]
mode_color = mode_cfg["color"]

# Header del modo actual
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid #1e293b;margin-bottom:16px">
  <span style="font-size:1.8rem">{mode_cfg['icon']}</span>
  <div>
    <div style="color:#f8fafc;font-size:1.1rem;font-weight:600">Modo {mode_cfg['label']}</div>
    <div style="color:#64748b;font-size:0.78rem">
      {'Especificaciones exactas y procedimientos técnicos' if st.session_state.mode == 'tecnico' else
       'Recomendaciones comerciales y perfil de cliente' if st.session_state.mode == 'ventas' else
       'Diagnóstico de problemas y selección de equipo'}
    </div>
  </div>
  {'<span style="margin-left:auto;background:#1e293b;border:1px solid #334155;color:#94a3b8;font-size:0.7rem;padding:4px 10px;border-radius:20px;font-family:IBM Plex Mono">Multi-producto</span>' if mode_cfg['multi'] else ''}
</div>
""", unsafe_allow_html=True)

# Historial
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        badge_color = MODE_CONFIG[msg.get("mode", "tecnico")]["color"]
        badge_label = MODE_CONFIG[msg.get("mode", "tecnico")]["label"]
        prod_tags = "".join(f'<span class="product-tag">{p}</span>'
                            for p in msg.get("products", []))
        st.markdown(
            f'<div class="msg-assistant" style="--resp-color:{badge_color}">'
            f'<span class="msg-mode-badge" style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}44">'
            f'{badge_label}</span> {prod_tags}<br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )

# Input
with st.form("chat_form", clear_on_submit=True):
    col_q, col_btn = st.columns([6, 1])
    with col_q:
        placeholder_map = {
            "tecnico":     "Ej: ¿Cuál es la categoría CAT del CM4001?",
            "ventas":      "Ej: ¿A qué tipo de cliente le recomendarías este equipo?",
            "diagnostico": "Ej: Tengo disparos intermitentes en un diferencial, ¿qué equipo usar?",
        }
        user_input = st.text_input("", placeholder=placeholder_map[st.session_state.mode],
                                    label_visibility="collapsed")
    with col_btn:
        submitted = st.form_submit_button("→", type="primary", use_container_width=True)

# Preguntas rápidas
st.markdown("**Preguntas rápidas:**")
quick_cols = st.columns(2)
quick_qs = QUICK_QUESTIONS[st.session_state.mode]
for i, q in enumerate(quick_qs):
    with quick_cols[i % 2]:
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            user_input = q
            submitted = True


# ── Procesar pregunta ─────────────────────────────────────────────────────────
if submitted and user_input and user_input.strip() and index:
    st.session_state.messages.append({"role": "user", "content": user_input})

    mode = st.session_state.mode
    is_multi = MODE_CONFIG[mode]["multi"]
    pf = st.session_state.product_filter

    # Búsqueda
    stats = index.search_with_stats(
        user_input, top_k=6,
        product_filter=pf,
        multi_product=(is_multi and pf is None)
    )
    results = stats["results"]
    st.session_state.last_sources = results

    max_chars = 1400 if cpu_mode else MAX_CTX_CHARS
    context = format_context(results, max_chars)

    # Mostrar productos encontrados
    prods_found = stats.get("products_found", [])
    if prods_found:
        prod_html = " ".join(f'<span class="product-tag">{p}</span>' for p in prods_found)
        pages = list(dict.fromkeys(f"p{r['page']}" for r in results))
        st.markdown(
            f'<div style="margin:4px 0 8px;font-size:0.75rem;color:#64748b">'
            f'Contexto: {prod_html} — {" · ".join(pages)}</div>',
            unsafe_allow_html=True
        )

    prompt = build_prompt(user_input, context, mode)

    with st.spinner(f"Modo {MODE_CONFIG[mode]['label']} — procesando..."):
        full_response = ""
        placeholder = st.empty()

        if ollama_ok:
            for token in query_ollama_stream(prompt, model_name):
                full_response += token
                placeholder.markdown(
                    f'<div class="msg-assistant" style="--resp-color:{mode_color}">'
                    f'{full_response}▌</div>',
                    unsafe_allow_html=True
                )
            placeholder.markdown(
                f'<div class="msg-assistant" style="--resp-color:{mode_color}">'
                f'{full_response}</div>',
                unsafe_allow_html=True
            )
        else:
            full_response = f"[Ollama no disponible] Contexto recuperado:\n\n{context[:600]}..."
            placeholder.markdown(
                f'<div class="msg-assistant">{full_response}</div>',
                unsafe_allow_html=True
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "mode": mode,
        "products": prods_found,
    })
    st.rerun()