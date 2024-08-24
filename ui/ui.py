import os.path

from flask import Flask
import common_functions as core

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CODE_DIR, 'static')
# https://flask.palletsprojects.com/en/2.3.x/patterns/fileuploads/
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ALLOWED_EXTENSIONS = {'png'}

app = Flask(__name__, static_url_path='', static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def root():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(debug=True)

# File upload/download features are based on following references
# https://pythonbasics.org/flask-upload-file/
# https://stackoverflow.com/questions/2368784/draw-on-html5-canvas-using-a-mouse
# https://stackoverflow.com/questions/10906734/how-to-upload-image-into-html5-canvas
# https://stackoverflow.com/questions/13198131/how-to-save-an-html5-canvas-as-an-image-on-a-server
# https://stackoverflow.com/questions/8126623/downloading-canvas-element-to-an-image
