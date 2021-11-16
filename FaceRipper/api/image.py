import pdb
import json
import flask
import imageio
import FaceRipper
import time
import uuid
import pathlib
from FaceRipper.api.model import get_pred, drawMask
import numpy as np

@FaceRipper.app.route("/", methods=["get"])
def index():
    return flask.render_template("index.html")

@FaceRipper.app.route("/api/image/mask/", methods=["post"])
def mask_api():
    input_img = flask.request.files['image']
    img = imageio.imread(input_img)
    #start = time.time()
    pred = get_pred(img)
    #end = time.time()
    pred = pred[0]
    maskedImg = drawMask(img, pred)
    filename = save_file(maskedImg, input_img.filename)
    context = {
        "result" : filename
    }
    return flask.render_template("show_img.html", **context)

@FaceRipper.app.route('/result/<filename>')
def get_result(filename):
    """Route for display uploads."""
    try:
        return flask.send_from_directory(
            FaceRipper.config.TMP_DIR, filename), 200
    except FileNotFoundError:
        return flask.abort(404)

# file io ================================================
def save_file(file, filename):
    """Help save file."""
    # Unpack flask object
    stem = uuid.uuid4().hex
    suffix = pathlib.Path(filename).suffix
    uuid_basename = f"{stem}{suffix}"
    # Save to disk
    path = FaceRipper.config.TMP_DIR/uuid_basename
    imageio.imwrite(path, file)
    return uuid_basename
import shutil

def cleanup_thread(signal):
    """Help cleanup file."""
    filepath = FaceRipper.config.TMP_DIR
    while not signal["shutdown"]:
        shutil.rmtree(filepath)
        time.sleep(600)