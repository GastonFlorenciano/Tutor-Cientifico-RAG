import streamlit as st
import rag_core
import time

# --- 1. Configuración de la Página ---
st.set_page_config(
    page_title="Tutor de IA",
    page_icon="🤖",
    layout="centered"
)

# --- 2. Título y Descripción ---
st.title("🤖 Tutor de Investigación de IA")
st.markdown("Chatea con los papers fundacionales (Attention, BERT, RAG) usando **Mistral** via Ollama.")

# --- 3. Inicialización del Historial de Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Barra Lateral ---
with st.sidebar:
    st.subheader("⚙️ Opciones")
    st.markdown("**Modelo:** Mistral (Ollama)")
    st.markdown("**Embedding:** nomic-embed-text")
    
    if st.button("🗑️ Limpiar Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("""
    ### ℹ️ Acerca de
    - **LLM:** Mistral (7B)
    - **Embedding:** nomic-embed-text
    - **Vector DB:** ChromaDB
    - **Papers:** Attention, BERT, RAG
    
    ### 💡 Tips
    - Haz preguntas sobre Transformers, BERT o RAG
    - Sé específico en tus preguntas
    - Las respuestas pueden tardar 30-60 segundos
    """)

# --- 5. Mostrar Mensajes Antiguos ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. Lógica del Chat ---
if prompt := st.chat_input("¿Qué pregunta tienes sobre los papers?"):
    
    # Añadir y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta del asistente
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with placeholder.container():
            st.write("⏳ Procesando tu pregunta...")
            progress_bar = st.progress(0)
            
            # Simular progreso mientras se procesa
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        
        try:
            start_time = time.time()
            response = rag_core.RAG_CHAIN_GLOBAL.invoke(prompt)
            elapsed_time = time.time() - start_time
            
            with placeholder.container():
                st.markdown(response)
                st.caption(f"⏱️ Tiempo de respuesta: {elapsed_time:.1f}s")
        except Exception as e:
            response = f"❌ Error: {str(e)}"
            with placeholder.container():
                st.error(response)
    
    # Guardar la respuesta
    st.session_state.messages.append({"role": "assistant", "content": response})