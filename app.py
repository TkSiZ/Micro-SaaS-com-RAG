import streamlit as st
from src.llm import generate_answer

st.title("Tutor de Teoria Musical")
st.caption("Faça perguntas com base nos materiais indexados.")

if "historico" not in st.session_state:
    st.session_state.historico = []

pergunta = st.chat_input("Ex: O que é uma cadência plagal?")

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando os materiais..."):
            resposta = generate_answer(pergunta)
        st.write(resposta)
    
    st.session_state.historico.append({"role": "assistant", "content": resposta})