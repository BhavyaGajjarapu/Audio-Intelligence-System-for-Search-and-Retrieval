from flask import Flask, render_template, request, redirect, url_for
from main import semantic_search_audio_catalog, load_catalog  
import os
import json
from main import process_audio  # This should be your audio processing logic

AUDIO_FOLDER = 'audio_folder'  # Folder where audios are stored
CATALOG_PATH = 'audio_catalog.json'

app = Flask(__name__)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def load_catalog():
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            return json.load(f)
    return []

def save_catalog(catalog):
    with open(CATALOG_PATH, 'w') as f:
        json.dump(catalog, f, indent=2)

def preprocess_audios():
    catalog = load_catalog()
    existing_files = {audio['file_name'] for audio in catalog}
    for fname in os.listdir(AUDIO_FOLDER):
        if fname.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')) and fname not in existing_files:
            fpath = os.path.join(AUDIO_FOLDER, fname)
            meta = process_audio(fpath)
            catalog.append(meta)
            save_catalog(catalog)
    return catalog

def process_and_add_audio(filepath, filename):
    meta = process_audio(filepath)
    catalog = load_catalog()
    if not any(audio['file_name'] == filename for audio in catalog):
        catalog.append(meta)
        save_catalog(catalog)

@app.route('/', methods=['GET', 'POST'])
def index():
    preprocess_audios()  # Ensure all files in folder are processed
    catalog = load_catalog()
    message = None

    # Handle upload
    if request.method == 'POST' and 'audiofile' in request.files and request.files['audiofile'].filename:
        file = request.files['audiofile']
        save_path = os.path.join(AUDIO_FOLDER, file.filename)
        if not os.path.exists(save_path):
            file.save(save_path)
            process_and_add_audio(save_path, file.filename)
            message = f"Uploaded and processed: {file.filename}"
        else:
            message = f"File '{file.filename}' already exists. Using existing data."
        catalog = load_catalog()

    # Handle search (GET or POST)
    results = []
    query = ""
    if request.method == 'POST' and 'query' in request.form:
        query = request.form['query'].strip()
    elif request.method == 'GET':
        query = request.args.get('query', '').strip()
    if query:
        results = semantic_search_audio_catalog(query, catalog)

    return render_template('index.html', catalog=catalog, message=message, results=results, query=query)

# Example for /search route
@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').strip()
    catalog = load_catalog()  # Load your catalog as a list of dicts
    results = semantic_search_audio_catalog(query, catalog)
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run(debug=True)