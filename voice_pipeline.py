from faster_whisper import WhisperModel
from rag import retrieve
from llm import generate_answer
from tts import text_to_speech
from pathlib import Path
import time


print("Loading Whisper model...")
whisper_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)
print("Whisper loaded.")


def transcribe_audio(audio_file):
    segments, info = whisper_model.transcribe(audio_file)

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text.strip()


def build_context(results):
    return "\n\n".join(
        result["text"]
        for result in results
    )


def process_voice(audio_file):

    print("\n🎤 Transcribing...")

    question = transcribe_audio(audio_file)

    if not question:
        return None, None, None, []


    print(f"Question: {question}")


    print("\n🔎 Searching FAQ...")

    results = retrieve(question)

    context = build_context(results)


    print("\n🧠 Generating answer...")

    answer = generate_answer(
        question,
        context
    )

    print(f"Answer: {answer}")


    print("\n🔊 Generating voice...")

    # Create a unique filename for every answer
    timestamp = int(time.time() * 1000)

    output_filename = f"answer_{timestamp}.wav"

    audio_output = text_to_speech(
        answer,
        output_filename
    )

    print(f"Audio generated: {audio_output}")


    return (
        question,
        answer,
        audio_output,
        results
    )