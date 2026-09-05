import streamlit as st
import tempfile
from pathlib import Path

from voice_pipeline import process_voice


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Campus Voice AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

def get_confidence(results):
    if not results:
        return "🔴 No Match", 0.0

    scores = [
        float(result.get("score", 0))
        for result in results
    ]

    best_score = max(scores)

    if best_score >= 0.50:
        return "🟢 High Confidence", best_score

    elif best_score >= 0.30:
        return "🟡 Moderate Confidence", best_score

    else:
        return "🔴 Low Confidence", best_score

# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #ddd;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎓 Campus Portal")
    st.caption("AI Student Helpdesk")

    st.divider()

    st.subheader("📚 FAQ Topics")

    st.write("• General Information")
    st.write("• Admissions")
    st.write("• Examination")
    st.write("• Attendance")
    st.write("• Certificates")
    st.write("• Library")
    st.write("• Student Support")

    st.divider()

    st.subheader("⚙️ AI Pipeline")

    st.write("🎤 Speech-to-Text")
    st.write("🔎 FAQ Retrieval")
    st.write("🧠 Local LLM")
    st.write("🔊 Text-to-Speech")

    st.divider()

    st.success("🟢 AI Assistant Online")

    st.caption(
        "Whisper + RAG + Ollama + Piper"
    )


# =========================================================
# HEADER
# =========================================================

st.title("🎓 Campus Voice AI")

st.subheader("Your AI-powered college helpdesk")

st.success(
    "Ask college-related questions using your voice. "
    "The assistant searches the college knowledge base "
    "and gives a grounded spoken answer."
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def process_recording(audio):

    temp_audio = None

    try:

        # Save recording temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(audio.getvalue())
            temp_audio = temp_file.name


        # Process complete pipeline
        question, answer, audio_output, results = (
            process_voice(temp_audio)
        )


        # Check whether speech was recognized
        if not question:

            st.warning(
                "🎤 I couldn't understand the recording. "
                "Please speak clearly and try again."
            )

            return


        # Check whether answer was generated
        if not answer:

            st.warning(
                "🤖 I couldn't generate an answer. "
                "Please try asking the question again."
            )

            return


        # Save conversation
        st.session_state.conversation.append(
            {
                "question": question,
                "answer": answer,
                "audio": (
                    str(audio_output)
                    if audio_output
                    else None
                ),
                "results": results
            }
        )

        st.session_state.question_count += 1

        # Refresh UI
        st.rerun()


    except Exception as e:

        error_message = str(e).lower()


        # Ollama-related error
        if (
            "connection" in error_message
            or "11434" in error_message
            or "ollama" in error_message
        ):

            st.error(
                "🧠 **AI model is unavailable.**\n\n"
                "Please make sure Ollama is running and "
                "the `llama3.2:3b` model is available."
            )


        # Whisper/audio error
        elif (
            "whisper" in error_message
            or "audio" in error_message
            or "ffmpeg" in error_message
        ):

            st.error(
                "🎤 **Audio processing failed.**\n\n"
                "Please record your question again and "
                "make sure your microphone is working."
            )


        # TTS error
        elif (
            "piper" in error_message
            or "tts" in error_message
            or ".wav" in error_message
        ):

            st.error(
                "🔊 **Voice generation failed.**\n\n"
                "The text answer may still be available, "
                "but the spoken response could not be generated."
            )


        # General error
        else:

            st.error(
                "❌ **Something went wrong.**\n\n"
                "Please try recording your question again."
            )

            # Developer information
            with st.expander(
                "🔧 Technical details"
            ):

                st.code(str(e))


    finally:

        # Always remove temporary recording
        if temp_audio:

            try:
                Path(temp_audio).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.conversation:

    st.info(
        "👋 Welcome! Record your question below and click "
        "**➡️ Send Question**."
    )

    st.write("### ✨ What can I ask?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**📚 Academic**\n\n"
            "Exams, attendance and academic support."
        )

    with col2:

        st.info(
            "**📝 Administration**\n\n"
            "Admissions, certificates and college office."
        )

    with col3:

        st.info(
            "**📖 Student Services**\n\n"
            "Library and student services."
        )

    st.write("### 💡 Try asking")

    st.write(
        "• What are the college working hours?\n\n"
        "• How can I get a bonafide certificate?\n\n"
        "• Where can I find the examination timetable?\n\n"
        "• How can I borrow a library book?"
    )

    st.divider()


# =========================================================
# FIRST RECORDING
# =========================================================

if not st.session_state.conversation:

    st.write("### 🎤 Ask your first question")

    audio = st.audio_input(
        "🎤 Record your question",
        key="first_recorder"
    )

    if audio:

        st.audio(audio)

        send = st.button(
            "➡️ Send Question",
            key="first_send",
            type="primary",
            use_container_width=True
        )

        if send:

            with st.spinner(
                "🎤 Understanding your voice..."
            ):

                process_recording(audio)

# =========================================================
# CONVERSATION HISTORY
# =========================================================

if st.session_state.conversation:

    st.write("### 💬 Conversation")

    for index, item in enumerate(
        st.session_state.conversation
    ):

        # USER MESSAGE
        with st.chat_message(
            "user",
            avatar="🧑‍🎓"
        ):

            st.write(item["question"])


        # AI MESSAGE
        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            st.write(item["answer"])

            # Confidence
            confidence_label, confidence_score = get_confidence(
                item.get("results", [])
            )

            st.caption(
                f"{confidence_label} • "
                f"FAQ match: {confidence_score:.3f}"
            )

            # Audio answer
            if item.get("audio"):

                audio_path = Path(
                    item["audio"]
                )

                if audio_path.exists():

                    st.audio(
                        str(audio_path),
                        format="audio/wav"
                    )

            # RAG sources
            results = item.get(
                "results",
                []
            )

            if results:

                with st.expander(
                    f"🔎 View RAG Sources ({len(results)})"
                ):

                    st.caption(
                        "FAQ entries retrieved from the "
                        "college knowledge base."
                    )

                    for source_index, result in enumerate(
                        results,
                        start=1
                    ):

                        st.markdown(
                            f"**🔎 Source {source_index}**"
                        )

                        score = result.get("score")

                        if score is not None:

                            st.progress(
                                min(
                                    max(float(score), 0.0),
                                    1.0
                                )
                            )

                            st.caption(
                                f"Relevance score: "
                                f"{float(score):.3f}"
                            )

                        st.write(
                            result.get(
                                "text",
                                "FAQ information unavailable."
                            )
                        )

                        if source_index < len(results):
                            st.divider()
# =========================================================
# NEW QUESTION
# =========================================================

if st.session_state.conversation:

    st.divider()

    st.write("### 🎤 Ask another question")

    new_audio = st.audio_input(
        "🎤 Record your question",
        key=f"recorder_{st.session_state.question_count}"
    )

    if new_audio:

        st.audio(new_audio)

        send = st.button(
            "➡️ Send Question",
            key=f"send_{st.session_state.question_count}",
            type="primary",
            use_container_width=True
        )

        if send:

            with st.spinner(
                "🎤 Understanding your voice..."
            ):

                process_recording(new_audio)


# =========================================================
# NEW CONVERSATION
# =========================================================

if st.session_state.conversation:

    st.divider()

    if st.button(
        "🔄 Start New Conversation",
        use_container_width=True
    ):

        st.session_state.conversation = []
        st.session_state.question_count = 0

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Campus Voice AI • College FAQ Assistant • "
    "Whisper + RAG + Ollama + Piper"
)