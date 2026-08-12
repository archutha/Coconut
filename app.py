import streamlit as st
from agent import ask_coconut

SAMPLE_QUESTIONS = [
    "What festivals happen in Spain during summer?",
    "Best sunscreen for oily skin in summer?",
    "Top summer travel destinations in Europe",
    "What's trending in summer fashion this year?",
]

st.set_page_config(
    page_title="Coconut - your Summer Assistant",
    page_icon="🥥",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.title("🥥 Coconut")
    st.caption("Your summer tourism, festivals & fashion assistant")
    st.divider()

    st.markdown("**Try asking**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []

st.title("Coconut")
st.caption("Ask me about summer tourism, festivals, or fashion.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask Coconut something...")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Coconut is thinking..."):
            response = ask_coconut(question)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
      