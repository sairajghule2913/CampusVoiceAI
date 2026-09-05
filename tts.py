import sys
import subprocess
from pathlib import Path

VOICE = "en_US-lessac-medium"
OUTPUT_DIR = Path("audio")
OUTPUT_DIR.mkdir(exist_ok=True)


def text_to_speech(text, output_file="answer.wav"):
    output_path = OUTPUT_DIR / output_file

    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper",
            "-m",
            VOICE,
            "-f",
            str(output_path),
            "--",
            text,
        ],
        check=True,
    )

    return output_path


if __name__ == "__main__":
    text = input("Enter text to speak: ")

    output = text_to_speech(text)

    print(f"\nAudio generated: {output}")