# app.py
# Este es el FRONTEND de tu aplicación.
# Usa Streamlit para crear la interfaz de chat.

import streamlit as st
import rag_core  # Importa nuestro "motor" (backend)

# --- 1. Configuración de la Página ---
# (Esto le da un título y un ícono a la pestaña del navegador)
st.set_page_config(
    page_title="Tutor de IA",
    page_icon="🤖",
    layout="centered"
)

# --- 2. Título y Descripción ---
st.title("🤖 Tutor de Investigación de IA")
st.markdown("Chatea con los papers fundacionales (Attention, BERT, RAG).")

# --- 3. Inicialización del Historial de Chat ---
# Streamlit necesita "session_state" para recordar la conversación.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Botón de Limpieza --- (Requisito del PDF [cite: 44])
# Colocamos un botón en la barra lateral
with st.sidebar:
    st.subheader("Opciones")
    if st.button("Limpiar Chat"):
        st.session_state.messages = []
        st.rerun() # Refresca la página

# --- 5. Mostrar Mensajes Antiguos ---
# Itera sobre el historial guardado y lo muestra en la interfaz.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. Lógica del Chat (Input y Respuesta) ---
# st.chat_input() crea la barra de chat en la parte inferior.
if prompt := st.chat_input("¿Qué es la 'atención' en un Transformer?"):
    
    # 1. Añadir y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generar y mostrar la respuesta del asistente (RAG)
    with st.chat_message("assistant"):
        # Usamos el spinner para mostrar que "está pensando"
        with st.spinner("Pensando..."):
            
            # ¡AQUÍ ES DONDE LLAMAMOS AL BACKEND!
            # Usamos la variable RAG_CHAIN_GLOBAL que creamos en rag_core.py
            response = rag_core.RAG_CHAIN_GLOBAL.invoke(prompt)
            
            st.markdown(response)
    
    # 3. Guardar la respuesta del asistente en el historial
    st.session_state.messages.append({"role": "assistant", "content": response})