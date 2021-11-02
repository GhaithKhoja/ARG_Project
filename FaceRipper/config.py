from pathlib import Path

APPLICATION_ROOT = Path('/')

MODEL_DIR = Path('FaceRipper/model') #directory that store mask rnn model

TMP_DIR = APPLICATION_ROOT/'tmp' # store temporary user image, clean up periodically