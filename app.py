from flask import Flask, render_template, request
import os
import whisper

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = whisper.load_model("base")

@app.route("/", methods=["GET", "POST"])
def index():
    transcription = ""
    segments = []
    if request.method == "POST":
        file = request.files.get("audio_file")
        language = request.form.get("language", "en")

        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            result = model.transcribe(filepath, language=language, verbose=False)
            transcription = result["text"]
            segments = result["segments"]

    return render_template("index.html", transcription=transcription, segments=segments)

if __name__ == "__main__":
    app.run(debug=True)
