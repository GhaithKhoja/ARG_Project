"""Face Ripper REST API."""

import flask

app = flask.Flask(__name__)

app.config.from_object('FaceRipper.config')

import FaceRipper.api
