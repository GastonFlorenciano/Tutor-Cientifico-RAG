# 🤖 Tutor de Investigación Científica de IA

**Proyecto Final para la materia:** S.A.C. - Modelos y Aplicaciones de la
Inteligencia Artificial.

**Instituto:** IPF "Dr. Alberto Marcelo Zerrilla"

Este proyecto es una solución de software basada en IA utilizando la
arquitectura RAG (Retrieval-Augmented Generation) para actuar como un "Tutor de
Investigación" experto.

Su base de conocimiento se compone exclusivamente de tres papers fundacionales
de la IA moderna:

- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding" (Devlin et al., 2018)
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et
  al., 2020)

---

### 🎯 El Problema y la Solución (Caso de Uso)

- **El Problema:** Un estudiante necesita consultar conceptos técnicos complejos
  de los papers fundacionales (ej. "¿Qué es la 'atención' en un Transformer?").
  El sistema debe ser capaz de explicar estos conceptos basándose exclusivamente
  en el contenido de los tres papers.

- **La Solución:** Este asistente de chat, construido con arquitectura RAG,
  ingiere los tres papers fundacionales. El sistema recupera el contexto más
  relevante para responder preguntas en lenguaje natural sobre los fundamentos
  de las arquitecturas Transformer, BERT y RAG.

- **Requisito Cumplido:** Al generar una respuesta, el sistema debe citar cuál
  de los tres papers utilizó como fuente principal para esa respuesta.

---

### 🛠️ Stack Tecnológico

- **Arquitectura:** RAG (Retrieval-Augmented Generation)
- **Framework RAG:** LangChain (utilizando LCEL)
- **Base de Datos Vectorial:** ChromaDB (Persistente)
- **Modelo de Embedding:** Ollama (`nomic-embed-text`)
- **Modelo de Lenguaje (LLM):** Ollama (`mistral` - 7B)
- **Frontend:** Streamlit
- **Lenguaje:** Python

---

### 📦 Instalación

Sigue estos pasos para clonar el repositorio e instalar todas las dependencias
necesarias.

**1. Clonar el Repositorio:**

```bash
git clone https://github.com/GastonFlorenciano/Tutor-Cientifico-RAG.git

# Ir al directorio del repositorio
cd Tutor-Cientifico-RAG
```

**2. Crear y Activar un Entorno Virtual:**

```bash
# Crear el entorno
python -m venv venv

# Activar en Windows
.\venv\Scripts\activate

# Activar en macOS/Linux
source venv/bin/activate
```

**3. Instalar dependencias:**

```bash
pip install -r requirements.txt
```

**4. Configurar Modelos (Ollama):**

El proyecto utiliza **Ollama** de forma local para tanto el embedding como el
LLM.

- **Instalación de Ollama:**

  1. Descarga e instala [Ollama](https://ollama.com/download) en tu máquina
  2. En tu terminal, descarga los modelos necesarios:

     ```bash
     # Modelo de embedding
     ollama pull nomic-embed-text

     # Modelo de lenguaje
     ollama pull mistral
     ```

  3. Verifica que los modelos están descargados:
     ```bash
     ollama list
     ```

- **Variables de Entorno (.env):**

  1. Crea un archivo llamado **`.env`** en la raíz del proyecto
  2. Añade la siguiente configuración:

     ```text
     # Ollama Configuration
     OLLAMA_BASE_URL=http://localhost:11434
     LLM_MODEL=mistral
     EMBEDDING_MODEL=nomic-embed-text

     # ChromaDB Configuration
     CHROMA_PERSIST_DIRECTORY=./chroma_db_tutor_ia
     COLLECTION_NAME=ia_papers_tutor

     # Data Configuration
     DATA_FOLDER=./docs
     ```

---

### ▶️ Instrucciones de Uso

**1. Asegúrate que Ollama está corriendo:**

En una terminal, ejecuta:

```bash
ollama serve
```

Deberías ver:

```
2025-11-16 20:15:00 - Ollama is running on http://localhost:11434
```

**2. En otra terminal, ejecuta el frontend:**

```bash
streamlit run app.py
```

Deberías ver:

```
Local URL: http://localhost:8501
```

**3. Abre tu navegador:**

Ve a: **http://localhost:8501**

---

### 💡 Características

- ✅ **Basado únicamente en papers:** Las respuestas provienen exclusivamente de
  los 3 papers fundacionales
- ✅ **Citas automáticas:** Cada respuesta incluye la fuente del paper utilizado
- ✅ **RAG Pipeline:** Recuperación inteligente de contexto relevante
- ✅ **Modelos locales:** Usa Ollama para privacidad y control total
- ✅ **Interfaz intuitiva:** Frontend Streamlit fácil de usar
- ✅ **Persistencia:** ChromaDB almacena los embeddings de forma persistente

---

### 📋 Requisitos del Sistema

- **Python:** 3.8 o superior
- **RAM:** Mínimo 8GB (recomendado 16GB)
- **Ollama:** Debe estar instalado y ejecutándose localmente
- **Almacenamiento:** ~5GB para los modelos de Ollama

---

### 🔧 Troubleshooting

**Error: "No se pudo conectar a Ollama"**

- Verifica que `ollama serve` está corriendo en otra terminal
- Comprueba que la URL en `.env` es correcta: `http://localhost:11434`

**Error: "Modelo no encontrado"**

- Ejecuta: `ollama pull mistral` y `ollama pull nomic-embed-text`
- Verifica con: `ollama list`

**Respuestas lentas**

- Es normal que Mistral tarde 30-60 segundos por respuesta
- Usa una pregunta más específica para mejores resultados

---

### 📝 Licencia

Este proyecto es para propósitos educativos.

---

### 👨‍💻 Autor

**Marcos López**

Estudiante de IPF "Dr. Alberto Marcelo Zerrilla"
