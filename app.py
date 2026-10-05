import sys
from pathlib import Path
import time
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag_pipeline import rag_answer


st.set_page_config(
    page_title="GenAI Knowledge Brain",
    page_icon="🧠",
    layout="wide"
)


st.title(
    "🧠 GenAI Knowledge Brain"
)

st.write(
    """
Ask questions about the local AI knowledge
base. Answers are generated using Retrieval
Augmented Generation with local embeddings,
ChromaDB and Ollama.
"""
)


with st.sidebar:

    st.header(
        "RAG Settings"
    )

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=5,
        value=3
    )

    st.markdown(
        """
### Pipeline

Question  
↓  
Embedding  
↓  
Vector Search  
↓  
Relevant Chunks  
↓  
LLM  
↓  
Grounded Answer
"""
    )


question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: What is retrieval "
        "augmented generation?"
    )
)


if st.button(
    "Ask Knowledge Brain",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Enter a question first."
        )

    else:

        with st.spinner(
            "Retrieving knowledge..."
        ):

            start = time.perf_counter()

            try:

                result = rag_answer(
                    question,
                    top_k=top_k
                )

                latency = (
                    time.perf_counter()
                    - start
                )

                st.subheader(
                    "Answer"
                )

                st.write(
                    result["answer"]
                )

                col1, col2 = st.columns(
                    2
                )

                col1.metric(
                    "Retrieved Chunks",
                    len(
                        result["sources"]
                    )
                )

                col2.metric(
                    "Latency",
                    f"{latency:.2f}s"
                )

                st.subheader(
                    "Retrieved Evidence"
                )

                for number, item in enumerate(
                    result["sources"],
                    start=1
                ):

                    with st.expander(
                        f"Source {number} — "
                        f"{item['source']}"
                    ):

                        st.write(
                            "**Chunk:**",
                            item[
                                "chunk_number"
                            ]
                        )

                        st.write(
                            "**Vector Distance:**",
                            round(
                                item[
                                    "distance"
                                ],
                                4
                            )
                        )

                        st.write(
                            item["text"]
                        )

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )