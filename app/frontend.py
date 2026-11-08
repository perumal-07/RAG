
import requests
import streamlit as st


# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Retrieval augmented generation",
    page_icon="🤖",
    layout="wide"
)


# 2. APPLICATION CONFIGURATION

st.title("Retrieval augmented generation")

st.caption(
    "An intelligent retrieval augmented generation interface "
    "powered by Ollama and ChromaDB."
)

API_TIMEOUT = 300
API_URL = "http://127.0.0.1:8000"


# 3. SIDEBAR

with st.sidebar:

    # Document upload section

    st.subheader("📄 Knowledge Base")

    uploaded_document = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt", "md", "csv", "json", "html", "htm", "rtf"]
    )

    if st.button(
        "Upload and Process Document",
        disabled=uploaded_document is None,
        use_container_width=True
    ):

        if uploaded_document is not None:

            file_data = uploaded_document.getvalue()

            if len(file_data) > 10 * 1024 * 1024:

                st.error("Documents must be 10 MB or smaller.")

            else:

                try:

                    with st.spinner(
                        "Processing document and generating embeddings..."
                    ):

                        response = requests.post(
                            f"{API_URL}/upload",
                            files={
                                "file": (
                                    uploaded_document.name,
                                    file_data,
                                    uploaded_document.type or "application/octet-stream"
                                )
                            },
                            timeout=API_TIMEOUT
                        )

                        response.raise_for_status()

                        result = response.json()

                    st.success("Document uploaded and indexed successfully!")

                    st.write(
                        "Chunks indexed:",
                        result["chunks_indexed"]
                    )

                    st.write(
                        "File:",
                        result["filename"]
                    )

                except requests.RequestException as error:

                    st.error(
                        f"Document upload failed: {error}"
                    )

    st.divider()

    # Clear conversation button

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Welcome to retrieval augmented generation. "
                    "How can I help you today?"
                ),
                "sources": []
            }
        ]

        st.rerun()


# 4. INITIALIZE CHAT HISTORY

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to retrieval augmented generation.\n\n"
                "Upload documents and ask questions about their contents."
            ),
            "sources": []
        }
    ]


# 5. DISPLAY CHAT HISTORY

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# 6. CUSTOMER CHAT INPUT

question = st.chat_input(
    "Ask a question..."
)


# 7. SEND QUESTION TO FASTAPI

if question:

    # Display customer message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "sources": []
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate AI answer

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the uploaded documents..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question
                    },
                    timeout=API_TIMEOUT
                )

                response.raise_for_status()

                result = response.json()

                answer = result.get(
                    "answer",
                    "No answer received."
                )

                sources = result.get(
                    "sources",
                    []
                )

                # Display answer

                st.markdown(answer)

                # Save response in chat history

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except requests.RequestException as error:

                error_message = (
                    "Sorry, I couldn't connect to the "
                    "retrieval augmented generation service. "
                    "Please check whether FastAPI is running."
                )

                st.error(error_message)

                st.caption(f"Technical details: {error}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )
