import tensorflow as tf
from PIL import Image
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = np.array([np.array(Image.open("dataset/test/12463.jpg").resize((128, 128)), dtype=np.float32) / 255.0])

interpreter.set_tensor(input_details[0]["index"], image)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])

if output[0][0] <= 0.1:
    print("cat", end=" ")
else:
    print("no cat", end=" ")

print(output[0][0])