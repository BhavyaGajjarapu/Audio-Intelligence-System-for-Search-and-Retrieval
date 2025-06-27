import os, json, uuid, shutil
import whisper, torch, librosa
import numpy as np
from pydub import AudioSegment
from panns_inference import AudioTagging
from sentence_transformers import SentenceTransformer, util
import spacy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("C:/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe")
AudioSegment.ffprobe   = which("C:/ffmpeg-7.1.1-essentials_build/bin/ffprobe.exe")


# === Try loading diarization pipeline ===
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    diarization = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="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"
    )
except Exception as e:
    print(f"Diarization pipeline not loaded: {e}")
    diarization = None

# === Directories ===
UPLOAD_DIR, PROCESSED_DIR = "uploaded_audios", "processed_audios"
TRANSCRIPTS_DIR, METADATA_DIR = "transcripts", "metadata"
CATALOG_PATH = "catalog.json"
for d in [UPLOAD_DIR, PROCESSED_DIR, TRANSCRIPTS_DIR, METADATA_DIR]:
    os.makedirs(d, exist_ok=True)

# === Load models ===
model_embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
whisper_model = whisper.load_model("base")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
social_classifier = pipeline("text-classification", model="SkolkovoInstitute/roberta_toxicity_classifier")

# === Helper Functions ===
def save_upload(src):
    fid = str(uuid.uuid4())
    ext = os.path.splitext(src)[1].lower()
    dst = os.path.join(UPLOAD_DIR, fid + ext)
    shutil.copy(src, dst)
    return dst, fid, os.path.basename(src)

def convert(src, fid):
    audio = AudioSegment.from_file(src).set_frame_rate(16000).set_channels(1)
    out = os.path.join(PROCESSED_DIR, f"{fid}.wav")
    audio.export(out, format="wav")
    return out

def load_wave(path):
    y, sr = librosa.load(path, sr=16000)
    print(f"Loaded {path}: {len(y)/sr:.2f}s")
    return y, sr

def detect_audio_tags(path):
    import csv
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tagger = AudioTagging(checkpoint_path=None, device=device)
    waveform, _ = librosa.load(path, sr=32000, mono=True)
    waveform = waveform[None, :]
    result = tagger.inference(waveform)
    output = result[0] if isinstance(result, tuple) else result

    if isinstance(output, torch.Tensor):
        probs = output.detach().cpu().numpy()
    elif isinstance(output, np.ndarray):
        probs = output
    else:
        raise TypeError(f"Unexpected output type: {type(output)}")

    with open("C:/Users/Bhavya Gajjarapu/panns_data/class_labels_indices.csv") as f:
        reader = csv.DictReader(f)
        class_names = [r["display_name"] for r in reader]

    probs = probs[0] if probs.ndim > 1 else probs
    sorted_indices = np.argsort(probs)[::-1][:5]
    return [(class_names[i], float(probs[i])) for i in sorted_indices]

def transcribe(path):
    return whisper_model.transcribe(path, language="en", fp16=False)["text"]

def emotion_recognition(path):
    tags = detect_audio_tags(path)
    emo = [t for t, _ in tags if any(e in t.lower() for e in ["happy", "sad", "angry", "fear"])]
    return emo[0] if emo else "neutral"

def diarize(path):
    if diarization is None:
        print("Diarization not available. Skipping.")
        return []
    di = diarization({"audio": path})
    return [{"start": turn.start, "end": turn.end, "speaker": speaker} for turn, _, speaker in di.itertracks(yield_label=True)]

def analyze_text(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_tones_local(text):
    tones = {}
    try:
        emotion_preds = emotion_classifier(text[:512])[0]
        tones['emotions'] = sorted(emotion_preds, key=lambda x: x['score'], reverse=True)[:3]
    except Exception as e:
        tones['emotions'] = []
        print(f"Emotion analysis failed: {e}")
    try:
        social_preds = social_classifier(text[:512])
        tones['toxicity'] = social_preds
    except Exception as e:
        tones['toxicity'] = []
        print(f"Social tone analysis failed: {e}")
    return tones

def build_metadata(tags, transcript, emotion, diarise, entities, tones, file_id, fname):
    meta = {
        "file_id": file_id,
        "file_name": fname,
        "tags": [t for t, _ in tags],
        "tag_probs": {t: p for t, p in tags},
        "transcript": transcript,
        "emotion": emotion,
        "diarization": diarise,
        "entities": entities,
        "tones": tones,
        "topic_summary": []
    }
    with open(os.path.join(METADATA_DIR, f"{file_id}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def semantic_search_audio_catalog(query, catalog, threshold=0.4):
    """
    Returns a list of audios from catalog that semantically match the query.
    Each result includes a similarity score.
    """
    query_emb = model_embedder.encode(query, convert_to_tensor=True)
    results = []
    for item in catalog:
        # Combine tags and transcript for semantic search
        combined_text = " ".join(item.get("tags", [])) + " " + item.get("transcript", "")
        combined_emb = model_embedder.encode(combined_text, convert_to_tensor=True)
        similarity = util.cos_sim(query_emb, combined_emb).item()
        if similarity > threshold:
            item = dict(item)  # Make a copy so we can add score
            item["match_score"] = similarity
            results.append(item)
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results

def update_catalog(meta):
    cat = []
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            cat = json.load(f)
    cat.append(meta)
    with open(CATALOG_PATH, "w") as f:
        json.dump(cat, f, indent=2)


def load_catalog():
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            return json.load(f)
    return []


def process_audio(src):
    upath, fid, fname = save_upload(src)
    wav = convert(upath, fid)
    load_wave(wav)
    tags = detect_audio_tags(wav)
    transcript = transcribe(wav)
    emotion = emotion_recognition(wav)
    diarise = diarize(wav)
    entities = analyze_text(transcript)
    tones = analyze_tones_local(transcript)
    meta = build_metadata(tags, transcript, emotion, diarise, entities, tones, fid, fname)
    update_catalog(meta)
    return meta

def process_all_audios_in_folder(folder):
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
            fpath = os.path.join(folder, fname)
            print(f"Processing: {fpath}")
            process_audio(fpath)

'''# === Run Example ===
if __name__ == "__main__":
    # Replace with your audio file path
    audio_file = "test1.mp3"  # or .wav etc.
    if os.path.exists(audio_file):
        meta = process_audio(audio_file)
        print(json.dumps(meta, indent=2))
    else:
        print(f"File not found: {audio_file}")'''

if __name__ == "__main__":
    process_all_audios_in_folder("audio_folder")  