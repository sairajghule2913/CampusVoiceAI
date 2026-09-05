# 🎓 Campus Voice AI

AI-powered voice FAQ assistant for college students.

## Features

- 🎤 Voice-based questions
- 🗣️ Speech-to-Text using Faster-Whisper
- 🔎 RAG-based FAQ retrieval
- 🧠 Local LLM using Ollama and Llama 3.2
- 🛡️ Grounded answers to reduce hallucinations
- 🔊 Text-to-Speech using Piper
- 💬 Conversation history
- 🔎 RAG source display
- 📊 Retrieval confidence indicator

## Architecture

Voice Input  
↓  
Faster-Whisper Speech-to-Text  
↓  
FAQ Retrieval (RAG)  
↓  
Ollama Llama 3.2  
↓  
Piper Text-to-Speech  
↓  
Spoken Answer

## Tech Stack

Python, Streamlit, Faster-Whisper, Scikit-learn, Ollama, Llama 3.2, Piper TTS.

## Knowledge Base

The application uses `data/college_faq.txt` as its college knowledge base.

The assistant is instructed to answer only from the available FAQ information and avoid inventing unsupported college policies.

## AI Coding Assistance Disclosure

AI coding assistance was used during development for code generation, debugging, architecture suggestions, UI improvements, and documentation. The final project was reviewed and tested by the developer.

## Limitations

The current knowledge base contains sample college FAQ information. Answer quality depends on the available FAQ content and speech recognition quality.
