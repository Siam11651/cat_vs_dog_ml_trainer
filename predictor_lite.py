import tensorflow as tf
from PIL import Image
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(prog="predictor", description="Detect whether an image contains a cat or dog")

parser.add_argument("--model", help="Set path to model", type=str, default="model/model.tflite")
parser.add_argument("--image", help="Set path to image", type=str)

args = parser.parse_args()
interpreter = tf.lite.Interpreter(args.model)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
image = np.array([Image.open(args.image).resize((128, 128))], dtype=np.float32) / 255.0

interpreter.set_tensor(input_details["index"], image)
interpreter.invoke()

output = interpreter.get_tensor(output_details["index"])

print(output) # 0 means cat