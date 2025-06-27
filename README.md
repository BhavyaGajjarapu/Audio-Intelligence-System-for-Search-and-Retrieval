# 🎧 Audio Intelligence Pipeline

This project is a modular audio processing and retrieval system designed to handle large sets of audio files. It performs the following:

- Converts audio to machine-readable `.wav` format
- Transcribes speech using OpenAI's Whisper
- Tags environmental and musical sounds using PANNs
- Stores structured metadata (e.g., transcript, tags, duration)
- Supports tag-based and text-based search for retrieval

---

## 🚀 Features

- 📥 **Audio Ingestion**: Upload and normalize audio files.
- 🧠 **Transcription**: High-quality speech-to-text using Whisper.
- 🏷️ **Audio Tagging**: Detects over 500+ sound categories using PANNs.
- 🗃️ **Metadata Storage**: JSON-based and folder-based structured storage.
- 🔎 **Search Interface**: Search by keywords in transcript or detected tags.

---

## 📁 Project Structure

project_root/

├── uploaded_audios/ # Raw uploaded audio files

├── processed_audios/ # Normalized and converted .wav files

├── transcripts/ # Transcribed text in JSON format

├── metadata/ # Metadata including tags, duration, etc.

├── app.py # Main pipeline execution script

├── requirements.txt # Python dependencies

└── README.md # Project description and usage


## Install dependencies:

pip install -r requirements.txt


## 🧪 How to Use

Place your audio files in uploaded_audios/.

## Run the pipeline:

python app.py

## Outputs:

Normalized audio → processed_audios/

Transcript → transcripts/

Metadata with tags → metadata/

## 🔍 Search Example

You can write search utilities to:

Match keywords from the transcript

Match PANN tags such as "speech", "dog barking", "applause", etc.

## 🧠 Models Used

Whisper: For speech transcription

PANNs (Cnn14): For audio tagging

## 📌 TODO

 Add web interface

 Support session-based refinement search

 Integrate vector-based semantic search
