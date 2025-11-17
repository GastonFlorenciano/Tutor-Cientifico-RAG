import streamlit as st
import uuid
import rag_core

# ----------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------
st.set_page_config(
    page_title="Tutor de IA",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Tutor de Investigación de IA")
st.markdown("Chatea con los papers fundacionales (Attention, BERT, RAG).")

# ----------------------------------
# 2. ESTADO GLOBAL
# ----------------------------------
# conversations = { chat_id: { "messages": [], "title": "" } }
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


# ----------------------------------
# 3. FUNCIÓN PARA CREAR NUEVO CHAT
# ----------------------------------
def create_new_chat():
    new_id = str(uuid.uuid4())[:8]
    st.session_state.conversations[new_id] = {
        "messages": [],
        "title": ""  # Vacío hasta que el usuario pregunte algo
    }
    st.session_state.current_chat_id = new_id


# Si no hay chat aún, crear uno
if st.session_state.current_chat_id is None:
    create_new_chat()


# ----------------------------------
# 4. SIDEBAR (HISTORIAL)
# ----------------------------------
with st.sidebar:
    st.subheader("Historial de Chats")

    for chat_id, data in st.session_state.conversations.items():

        # Nombre del chat:
        # Si todavía no tiene título → mostrar "(Chat nuevo)"
        title = data["title"] if data["title"] else "(Chat nuevo)"

        if st.button(title, key=f"btn_{chat_id}"):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.markdown("---")

    if st.button("➕ Nuevo Chat"):
        create_new_chat()
        st.rerun()


# ----------------------------------
# 5. OBTENER EL CHAT ACTIVO
# ----------------------------------
chat_id = st.session_state.current_chat_id
chat_data = st.session_state.conversations[chat_id]
messages = chat_data["messages"]


# ----------------------------------
# 6. MOSTRAR MENSAJES ANTERIORES
# ----------------------------------
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ----------------------------------
# 7. INPUT DEL USUARIO
# ----------------------------------
if prompt := st.chat_input("Escribe tu mensaje..."):

    # Guardar el mensaje del usuario
    messages.append({"role": "user", "content": prompt})

    # Si es el primer mensaje → usarlo como título del chat
    if chat_data["title"] == "":
        chat_data["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = rag_core.RAG_CHAIN_GLOBAL.invoke(prompt)
            st.markdown(response)

    messages.append({"role": "assistant", "content": response})
