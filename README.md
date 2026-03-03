# 🔧 Pipeline PDF → Ollama | Manual HIOKI CM4001

Pipeline completo para procesar PDFs técnicos con LLMs locales usando Ollama.
Sin dependencias de APIs externas. 100% local y privado.

---

## 📦 Arquitectura

```
PDF
 │
 ▼
1_extract_pdf.py     → Extrae texto y tablas → chunks.json / tables.json
 │
 ▼
2_build_index.py     → Crea índice TF-IDF    → index.pkl
 │
 ▼
3_chat.py            → Chat Q&A + Extracción  → (respuestas / extracted_data.json)
    └── Ollama (llama3 / llama3.2)
```

---

## 🚀 Instalación

### 1. Instalar dependencias Python

```bash
pip install pdfplumber pymupdf
```

> Solo `pdfplumber` es estrictamente necesario. El resto del pipeline
> usa Python puro (sin chromadb, sin sentence-transformers).

### 2. Instalar y configurar Ollama

```bash
# Instalar Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar el modelo
ollama pull llama3.2

# Iniciar el servidor (si no corre automáticamente)
ollama serve
```

---

## 📋 Uso paso a paso

### Paso 1 — Extraer el PDF

```bash
python 1_extract_pdf.py manual_CM4001.pdf
```

Genera en `output/`:
- `chunks.json` — fragmentos de texto con metadatos de página y sección
- `tables.json` — tablas del manual en formato estructurado

### Paso 2 — Construir el índice

```bash
python 2_build_index.py
```

Genera:
- `output/index.pkl` — índice TF-IDF serializado

### Paso 3a — Chat interactivo

```bash
python 3_chat.py
# o especificando modelo:
python 3_chat.py --model llama3.2
```

Ejemplo de sesión:
```
👤 Tú: ¿Cuál es el rango de medición de corriente de fuga?
📚 Contexto: páginas p3, p6, p8
🤖  El instrumento puede medir corriente de fuga desde 0,60 mA 
    hasta 600,0 A, con los siguientes rangos manuales disponibles:
    AUTO → 60,00 mA → 600,0 mA → 6,000 A → 60,00 A → 600,0 A
```

Comandos dentro del chat:
| Comando | Acción |
|---------|--------|
| `fuentes` | Muestra las páginas usadas como contexto |
| `limpiar` | Limpia la pantalla |
| `salir` / `exit` | Termina el chat |

### Paso 3b — Extracción de datos estructurados

```bash
python 3_chat.py --extract
```

Extrae automáticamente estos campos del manual y los guarda en `output/extracted_data.json`:

| Campo | Descripción |
|-------|-------------|
| `nombre_producto` | Nombre y modelo del instrumento |
| `rango_medicion` | Rango mín/máx de corriente |
| `categoria_medicion` | Clasificación CAT y voltaje |
| `diametro_maximo` | Diámetro máximo de la mordaza |
| `funciones_principales` | Lista de capacidades principales |
| `comunicacion_inalambrica` | Info Bluetooth / Z3210 |
| `rangos_disponibles` | Todos los rangos de medición |

---

## ⚙️ Opciones avanzadas

```bash
# Usar otro modelo
python 3_chat.py --model llama3

# Aumentar contexto (más chunks = más preciso pero más lento)
python 3_chat.py --top-k 6

# Combinado
python 3_chat.py --extract --model llama3 --top-k 5
```

---

## 🗂️ Estructura de archivos

```
pipeline/
├── 1_extract_pdf.py      # Extractor PDF → JSON
├── 2_build_index.py      # Indexador TF-IDF
├── 3_chat.py             # Chat + extracción con Ollama
├── README.md             # Este archivo
└── output/               # Generado automáticamente
    ├── chunks.json       # Fragmentos de texto
    ├── tables.json       # Tablas extraídas
    ├── index.pkl         # Índice TF-IDF
    └── extracted_data.json  # Datos extraídos (modo --extract)
```

---

## 🔍 Cómo funciona el RAG

```
Pregunta del usuario
       │
       ▼
TF-IDF Search (index.pkl)
       │
       ▼
Top-4 chunks más relevantes
       │
       ▼
Prompt = SYSTEM + CONTEXTO + PREGUNTA
       │
       ▼
Ollama (llama3.2)
       │
       ▼
Respuesta fundamentada en el manual
```

El motor de búsqueda TF-IDF asigna puntajes a cada fragmento según
qué tan relevantes son sus términos para la pregunta, ponderando
por frecuencia inversa de documento (IDF).

---

## 🐛 Solución de problemas

**Ollama no responde**
```bash
# Verificar que corre
curl http://localhost:11434/api/tags

# Reiniciar
pkill ollama && ollama serve
```

**Modelo no encontrado**
```bash
ollama list           # ver modelos instalados
ollama pull llama3.2  # instalar llama3.2
ollama pull llama3    # o llama3 (más grande, más preciso)
```

**Respuestas lentas**
- Usa `llama3.2` (3B params) en lugar de `llama3` (8B params)
- Reduce `--top-k` a 2 o 3
- Aumenta memoria RAM disponible para Ollama

**Chunks de baja calidad**
- El PDF puede tener texto en imágenes (OCR necesario)
- Ajusta `CHUNK_SIZE` en `1_extract_pdf.py` (prueba con 300-800)

---

## 📚 Dependencias

| Librería | Uso | Instalación |
|----------|-----|-------------|
| `pdfplumber` | Extracción de texto y tablas | `pip install pdfplumber` |
| `ollama` (servidor) | LLM local | [ollama.ai](https://ollama.ai) |
| Python stdlib | `json`, `pickle`, `urllib`, `re` | Incluido |

> No se requieren: chromadb, sentence-transformers, langchain, OpenAI API.
