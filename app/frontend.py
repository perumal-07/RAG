
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

    if st.button(
        "🗑️ Clear Chat",
        width="stretch"
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

# 6. CHAT INPUT

submission = st.chat_input(
    "Ask a question or attach a document...",
    accept_file="multiple",
    file_type=["pdf", "docx", "txt", "md", "csv", "json", "html", "htm", "rtf"]
)


# 7. PROCESS CHAT INPUT

if submission:

    question = submission.text.strip()
    uploaded_documents = submission.files

    for uploaded_document in uploaded_documents:

        file_data = uploaded_document.getvalue()

        if len(file_data) > 10 * 1024 * 1024:
            st.error(
                f"{uploaded_document.name} is larger than 10 MB."
            )
            continue

        try:

            with st.status(
                f"Processing {uploaded_document.name}...",
                expanded=False
            ) as upload_status:

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
                response.json()

                upload_status.update(
                    label=f"{uploaded_document.name} is ready",
                    state="complete"
                )

        except requests.RequestException as error:

            upload_status.update(
                label=f"Could not process {uploaded_document.name}",
                state="error"
            )
            st.error(
                f"Document upload failed for {uploaded_document.name}: {error}"
            )

    if not question:
        st.stop()

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
