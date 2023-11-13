import tensorflow as tf
from PIL import Image
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(prog="predictor", description="Detect whether an image contains a cat or dog")

parser.add_argument("--model", help="Set path to model", type=str, default="model/model.keras")
parser.add_argument("--image", help="Set path to image", type=str)

args = parser.parse_args()
model = tf.keras.saving.load_model(args.model)
image = np.array([Image.open(args.image).resize((128, 128))], dtype=np.float32) / 255.0
output = model.predict(image)

print(output) # 0 means cat