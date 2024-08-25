import base64
import json
import os.path
import uuid
from datetime import datetime

from flask import Flask, request

import common_functions as core

DATA_IMAGE_HEADER = "data:image/png;base64,"

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CODE_DIR, 'static')
DATA_DIR = os.path.join(CODE_DIR, 'data')
PARENT_DIR = os.path.dirname(CODE_DIR)
# https://flask.palletsprojects.com/en/2.3.x/patterns/fileuploads/
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ALLOWED_EXTENSIONS = {'png'}

os.chdir(PARENT_DIR)
app = Flask(__name__, static_url_path='', static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def temp_image_file(prefix="I") -> str:
    """
    Create a temporary image filename and return it
    """
    current_date_time_str = datetime.now().strftime("%Y%m%d%H%M%S")[2:]
    uuid_str = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{current_date_time_str}_{uuid_str}.png".lower()
    return os.path.join(DATA_DIR, filename)


INFERENCE = core.Inference(temp_image_file)


@app.route("/gen", methods=["POST"])
def generate_image():
    data = request.get_json(force=True)
    task = data.get("task", None)
    image = data.get("image", None)
    if not task or not image:
        return json.dumps({"error": "No task or image", "image": "", "sheet": ""}), 400, {
            "ContentType": "application/json"}
    # gen-image is used to generate a single image from a pixel art
    # gen-sheet is used to generate a spritesheet
    if task not in ("gen-image", "gen-sheet-e1", "gen-sheet-e2"):
        return json.dumps({"error": "Invalid task", "image": "", "sheet": ""}), 400, {"ContentType": "application/json"}
    image_path = temp_image_file(prefix="I")
    output_image_path = temp_image_file(prefix="O")
    image = image[len(DATA_IMAGE_HEADER):]
    base64_decoded = base64.b64decode(image)
    with open(image_path, "wb") as f:
        f.write(base64_decoded)

    if task == "gen-image":
        INFERENCE.sketch_to_image(sketch_path=image_path, output_path=output_image_path)
        core.resize_to_256(output_image_path)
        with open(output_image_path, "rb") as f:
            b64_encoded = base64.b64encode(f.read())
        b64_encoded = DATA_IMAGE_HEADER + b64_encoded.decode("utf-8")
        return json.dumps({"image": b64_encoded, "error": "", "sheet": ""}), 200, {
            "ContentType": "application/json"}
    else:
        if task == "gen-sheet-e1":
            INFERENCE.create_sheet_e1(input_image=image_path, output_image=output_image_path)
        else:  # gen-sheet-e2
            INFERENCE.create_sheet_e2(input_image=image_path, output_image=output_image_path)
        with open(output_image_path, "rb") as f:
            b64_encoded = base64.b64encode(f.read())
        b64_encoded = DATA_IMAGE_HEADER + b64_encoded.decode("utf-8")
        return json.dumps({"error": "", "image": "", "sheet": b64_encoded}), 200, {
            "ContentType": "application/json"}


@app.route("/")
def root():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(debug=True)

# File upload/download/canvas-sketch features are based on following references
# https://pythonbasics.org/flask-upload-file/
# https://stackoverflow.com/questions/2368784/draw-on-html5-canvas-using-a-mouse
# https://stackoverflow.com/questions/10906734/how-to-upload-image-into-html5-canvas
# https://stackoverflow.com/questions/13198131/how-to-save-an-html5-canvas-as-an-image-on-a-server
# https://stackoverflow.com/questions/8126623/downloading-canvas-element-to-an-image
