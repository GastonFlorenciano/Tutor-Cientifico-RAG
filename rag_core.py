import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar Variables de Entorno
load_dotenv()

# Constantes del Proyecto
DATA_FOLDER = "./docs"
PERSIST_DIRECTORY = "./chroma_db_tutor_ia"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "ia_papers_tutor"
LLM_MODEL = "gemini-2.5-flash"

PAPERS = {
    "AIAYN.pdf": "Attention Is All You Need (Vaswani et al., 2017)",
    "BERT.pdf": "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
    "RAG.pdf": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)"
}

# Funciones de Carga y Splitting

def load_and_tag_pdfs(data_dir: str, paper_sources: dict):
    """Carga PDFs y añade metadatos de 'source'."""
    documents = []
    print(f"Iniciando carga desde: {data_dir}")
    for filename, source_title in paper_sources.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"ADVERTENCIA: No se encontró {file_path}")
            continue
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = source_title
            documents.extend(pages)
            print(f"-> Cargado: {filename}")
        except Exception as e:
            print(f"ERROR al cargar {file_path}: {e}")
    return documents

def split_documents(documents):
    """Divide los documentos en chunks."""
    print(f"Iniciando división de {len(documents)} páginas...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"-> Chunks creados: {len(chunks)}")
    return chunks

# Función de Creación del Pipeline

def get_rag_pipeline():
    """
    Ensambla y devuelve el pipeline RAG completo (rag_chain).
    """
    print("Conectando a Ollama (Embedding)...")
    try:
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"ERROR: No se pudo conectar a Ollama. ¿Está corriendo? {e}")
        return None

    print("Cargando/Creando Vector Store (ChromaDB)...")
    if not os.path.exists(PERSIST_DIRECTORY):
        print("Base de datos no encontrada. Creando una nueva...")
        docs = load_and_tag_pdfs(DATA_FOLDER, PAPERS)
        chunks = split_documents(docs)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )
        print("-> Base de datos creada y guardada.")
    else:
        print("Cargando base de datos existente.")
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )
        print("-> Base de datos cargada.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs_with_sources(docs):
        return "\n\n---\n\n".join(
            f"Fuente: {doc.metadata['source']}\nFragmento: {doc.page_content}"
            for doc in docs
        )

    template = """
    Eres un "Tutor de Investigación" experto en IA. Tu única fuente de conocimiento son los siguientes fragmentos de papers fundacionales.

    REGLAS ESTRICTAS:
    1. Responde la pregunta del estudiante basándote *única y exclusivamente* en el contexto proporcionado.
    2. Al final de tu respuesta, DEBES citar la fuente exacta del paper que usaste (ej: "Fuente: Attention Is All You Need (Vaswani et al., 2017)").
    3. Si el contexto no contiene la información para responder, DEBES responder exactamente: "Lo siento, no tengo información sobre eso en mis documentos fundacionales."

    ---
    CONTEXTO PROPORCIONADO:
    {context}
    ---

    PREGUNTA DEL ESTUDIANTE:
    {question}

    RESPUESTA DEL TUTOR:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    print("Conectando a Google VertexAI (LLM)...")
    llm = ChatVertexAI(
        model=LLM_MODEL,
        temperature=0.3
    )
    
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("-> Pipeline RAG creado exitosamente.")
    return rag_chain

# Inicialización
print("Iniciando rag_core.py...")
RAG_CHAIN_GLOBAL = get_rag_pipeline()
print("¡rag_core.py listo! El objeto 'RAG_CHAIN_GLOBAL' está preparado.")