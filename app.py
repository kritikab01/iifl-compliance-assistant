import streamlit as st
from groq import Groq

st.set_page_config(page_title="IIFL Compliance Assistant", layout="wide")
st.title("IIFL Gold Loan Compliance Assistant")
st.caption("Build 0 — deployment smoke test. Retrieval not wired yet.")

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing from Secrets.")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

q = st.text_input(
    "Ask a question",
    "What is the maximum LTV for a gold loan of Rs 2 lakh under RBI rules?",
)

if st.button("Ask"):
    with st.spinner("Thinking..."):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": q}],
                temperature=0,
            )
            st.write(r.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")
