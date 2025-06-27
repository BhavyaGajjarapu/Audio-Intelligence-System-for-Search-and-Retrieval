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

├── app.py # Flask app entry point

├── templates/

│ └── index.html # UI for upload & search

├── static/ # CSS, JS, icons (if any)

├── uploaded_audios/ # Raw audio uploads

├── processed_audios/ # Normalized .wav files

├── transcripts/ # Transcribed text

├── metadata/ # Audio metadata (tags, duration, etc.)


## Install dependencies:

- pip install -r requirements.txt

## Run the pipeline:

- python app.py

Then open http://127.0.0.1:5000 in your browser.

## 💡 How It Works

- Upload your audio file via the UI.

- Audio is converted to .wav using pydub.

- Speech is transcribed using Whisper.

- Audio is tagged using the PANNs Cnn14 model.

- Metadata is saved and used for future searches.

## Outputs:

Normalized audio → processed_audios/

Transcript → transcripts/

Metadata with tags → metadata/

## 🔍 Search Example

You can write search utilities to:

- Match keywords from the transcript

- Match PANN tags such as "speech", "dog barking", "applause", etc.

## 🧠 Models Used

- Whisper: For speech transcription

- PANNs (Cnn14): For audio tagging

## 📌 TODO

 - Support session-based refinement search

 - Integrate vector-based semantic search
