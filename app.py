from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pathlib import Path
import face_recognition
import numpy as np
import cv2
import uuid
import pickle

app = Flask(__name__)
CORS(app)

UPLOADS_FOLDER = Path("uploads")
CROPS_FOLDER = Path("face_crops")
DB_PATH = "face_db.pkl"

UPLOADS_FOLDER.mkdir(exist_ok=True)
CROPS_FOLDER.mkdir(exist_ok=True)

def crop_faces_and_save(image_path, number):
    img = face_recognition.load_image_file(image_path)
    boxes = face_recognition.face_locations(img)
    saved_files = []
    for i, (top, right, bottom, left) in enumerate(boxes):
        face = img[top:bottom, left:right]
        filename = f"{number}.jpg" if len(boxes) == 1 else f"{number}_{i}.jpg"
        crop_path = CROPS_FOLDER / filename
        cv2.imwrite(str(crop_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved_files.append(filename)
    return saved_files

def encode_image(path):
    img = face_recognition.load_image_file(path)
    boxes = face_recognition.face_locations(img)
    if not boxes:
        return None
    encodings = face_recognition.face_encodings(img, boxes)
    return encodings[0] if encodings else None

def rebuild_face_db():
    face_db = {}
    for img_path in CROPS_FOLDER.glob("*.jpg"):
        encoding = encode_image(img_path)
        if encoding is not None:
            face_db[img_path.stem] = encoding
    with open(DB_PATH, "wb") as f:
        pickle.dump(face_db, f)
    return face_db

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/report-missing', methods=['POST'])
def report():
    file = request.files.get("image")
    number = request.form.get("reporter_number")
    if not file or not number:
        return jsonify({"error": "Missing image or reporter_number"}), 400

    filename = f"{number}_{uuid.uuid4().hex[:5]}.jpg"
    save_path = UPLOADS_FOLDER / filename
    file.save(save_path)

    saved_crops = crop_faces_and_save(save_path, number)
    if not saved_crops:
        return jsonify({"error": "No face detected"}), 400

    rebuild_face_db()
    return jsonify({"success": True, "original": filename, "crops": saved_crops})

@app.route('/report-found', methods=['POST'])
def search():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Missing search image"}), 400

    rebuild_face_db()

    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if not boxes:
        return jsonify({"error": "No face detected in search image"}), 400

    encoding = face_recognition.face_encodings(rgb, boxes)[0]
    min_dist = 1.0
    best_match = None
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
    for name, db_encoding in face_db.items():
        dist = np.linalg.norm(encoding - db_encoding)
        if dist < min_dist:
            min_dist = dist
            best_match = name
    if min_dist < 0.6:
        return jsonify({"match": True, "number": best_match, "distance": round(min_dist, 3)})
    else:
        return jsonify({"match": False, "closest": best_match, "distance": round(min_dist, 3)})

if __name__ == "__main__":
    app.run(debug=True, port=8000)