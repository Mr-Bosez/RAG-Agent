"""
Streamlit UI: OTC medicine RAG chatbot.
Set GROQ_API_KEY (Groq, gsk_...) in `.env` - no keys in the UI.
Run: streamlit run app.py
"""
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values

import streamlit as st

from rag import MedicineRAG, answer_without_llm, build_rag_answer_prompt, stream_openai_chat

_ENV_FILE = Path(__file__).resolve().parent / ".env"
GROQ_BASE = "https://api.groq.com/openai/v1"


def _load_env_file(path: Path = _ENV_FILE, *, override: bool = True) -> None:
    """Load `.env` without assuming UTF-8 (Windows often saves as cp1252).

    `override=True` so values in this project's `.env` win over empty shell vars.
    """
    if not path.is_file():
        return
    raw = path.read_bytes()
    text: str | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    for key, val in dotenv_values(stream=StringIO(text)).items():
        if val is None:
            continue
        k = key.strip()
        if not k:
            continue
        if not override and k in os.environ:
            continue
        os.environ[k] = val.strip()


_load_env_file()


def _llm_key() -> str:
    return os.environ.get("GROQ_API_KEY", "").strip()


def _llm_routing(key: str) -> tuple[str, list[str], str]:
    """Return (base_url, model_ids, default_model)."""
    return (
        GROQ_BASE,
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "llama-3.3-70b-versatile",
    )


# page_icon: ASCII escape keeps file UTF-8-safe on disk (Streamlit reads as UTF-8).
st.set_page_config(page_title="OTC Medicine Assistant", page_icon="\U0001F48A", layout="wide")

DISCLAIMER = (
    "This tool uses your uploaded OTC reference PDF and an AI model for suggestions only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment."
)


@st.cache_resource
def get_rag() -> MedicineRAG:
    return MedicineRAG()


def reset_chat() -> None:
    """Reset chat history and user state without affecting cached resources."""
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Ask about symptoms (e.g. headache, allergy, cough) or a medicine name. "
                "I'll use your OTC medicines PDF as the only source."
            ),
        }
    ]
    st.session_state.is_blocked = False


def init_session() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Ask about symptoms (e.g. headache, allergy, cough) or a medicine name. "
                    "I'll use your OTC medicines PDF as the only source."
                ),
            }
        ]
    if "is_blocked" not in st.session_state:
        st.session_state.is_blocked = False


def main() -> None:
    init_session()

    st.title("OTC medicine assistant (RAG)")
    st.caption(DISCLAIMER)

    active_key = _llm_key()
    base_url, model_list, default_model = _llm_routing(active_key)

    with st.sidebar:
        st.subheader("Settings")
        st.caption("LLM key from `.env`: **GROQ_API_KEY** (Groq).")
        st.caption(f"Config file: `{_ENV_FILE}`")
        if _ENV_FILE.is_file():
            st.caption("Status: **key loaded**" if active_key else "Status: **no key** in `.env`")
        else:
            st.caption("Status: **no `.env` file** at the path above")
        if active_key:
            st.caption("Using: **Groq**")
        model = st.selectbox("Model", model_list, index=model_list.index(default_model))
        top_k = st.slider("Retrieved passages", min_value=3, max_value=12, value=6)

        if st.button("🔄 Reset Chat"):
            reset_chat()
            st.rerun()
        
        if st.button("Rebuild index from PDF"):
            with st.spinner("Rebuilding embeddings..."):
                r = get_rag()
                n = r.build_index(force=True)
            st.success(f"Indexed {n} chunks.")
            st.rerun()

        st.divider()
        st.markdown(
            "**How it works:** PDF text is split into chunks, embedded locally "
            "(sentence-transformers), and the most relevant chunks are sent to the LLM "
            "with your question."
        )

    if not active_key:
        st.warning(
            f"Set **GROQ_API_KEY** in `{_ENV_FILE}` (one line, no spaces "
            "around `=`), save, then refresh. Remove blank copies of those names from "
            "Windows environment variables if the key still does not load."
        )

    rag = get_rag()
    try:
        n_chunks = rag.ensure_index()
    except Exception as e:
        st.error(f"Could not load or build the index: {e}")
        st.stop()

    st.sidebar.metric("Indexed chunks", n_chunks)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Reset button near chat input area
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Reset", help="Reset chat and start over"):
            reset_chat()
            st.rerun()
    
    # Show blocked status if user is blocked
    if st.session_state.is_blocked:
        st.warning("⚠️ You have been blocked from receiving suggestions due to age requirements. Please click 'Reset' to start a new conversation.")
    
    if prompt := st.chat_input("Describe symptoms or ask about an OTC medicine..."):
        user_prompt = prompt.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Check if user is blocked - don't process further
        if st.session_state.is_blocked:
            with st.chat_message("assistant"):
                blocked_msg = "I'm sorry, but I cannot provide any medical suggestions. You have been blocked from accessing this service due to age requirements. Please click the **Reset** button to start a new conversation."
                st.markdown(blocked_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": blocked_msg})
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner("Searching the OTC reference..."):
                hits = rag.retrieve(user_prompt, k=top_k)
                context = rag.format_context(user_prompt, k=top_k)

            use_llm = bool(active_key)
            if not use_llm:
                text = answer_without_llm(user_prompt, hits)
                st.markdown(text)
                st.session_state.chat_history.append({"role": "assistant", "content": text})
            else:
                messages = build_rag_answer_prompt(
                    user_prompt,
                    context,
                    history=st.session_state.chat_history,
                )
                try:
                    stream = stream_openai_chat(
                        messages,
                        model=model,
                        api_key=active_key,
                        base_url=base_url,
                    )
                    full = st.write_stream(_stream_openai_chunks(stream))
                    
                    # Check if response contains AGE_BLOCKED
                    if full and "AGE_BLOCKED" in full.upper():
                        st.session_state.is_blocked = True
                        blocked_msg = "I'm sorry, but I cannot provide any medical suggestions. You have been blocked from accessing this service due to age requirements. Please click the **Reset** button to start a new conversation."
                        st.markdown(blocked_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": blocked_msg})
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": full or ""})
                except Exception as e:
                    fallback = (
                        f"LLM request failed ({e}). Showing retrieved passages instead.\n\n"
                        + answer_without_llm(user_prompt, hits)
                    )
                    st.markdown(fallback)
                    st.session_state.chat_history.append({"role": "assistant", "content": fallback})


def _stream_openai_chunks(stream):
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


if __name__ == "__main__":
    main()
