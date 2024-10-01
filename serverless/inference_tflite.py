import numpy as np
import requests
from PIL import Image
import io
from tflite_runtime.interpreter import Interpreter

# url = "http://bit.ly/mlbookcamp-pants"  # Replace with the actual URL of the image

classes = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt",
]


def get_image(url):
    """
    Downloads the image from a URL and returns it as a PIL Image.

    Args:
    - url (str): URL of the image to download.

    Returns:
    - image (PIL.Image): The downloaded image.
    """
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    return image


def preprocess_image(image, target_size=(299, 299)):
    """
    Resizes and normalizes the image.

    Args:
    - image (PIL.Image): The image to preprocess.
    - target_size (tuple): The target size for resizing.

    Returns:
    - input_data (np.array): Preprocessed image ready for model inference.
    """
    # Resize the image
    image = image.resize(target_size)

    # Convert image to a NumPy array and normalize by dividing by 175.5, then subtracting 1
    image = np.array(image) / 175.5 - 1

    # Add a batch dimension
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    return input_data


def load_model(model_path):
    """
    Loads the TensorFlow Lite model using tflite_runtime.Interpreter.

    Args:
    - model_path (str): Path to the TFLite model file.

    Returns:
    - interpreter (tflite_runtime.Interpreter): Loaded TFLite interpreter.
    """
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict(url):
    """
    Runs inference on the input data using the provided TFLite interpreter.

    Args:
    - interpreter (tflite_runtime.Interpreter): The loaded TFLite interpreter.
    - input_data (np.array): Preprocessed image data for inference.

    Returns:
    - predicted_label (str): The predicted class label.
    """
    image = get_image(url)
    input_data = preprocess_image(image)
    interpreter = load_model(model_path="clothing-model.tflite")
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output (probability distribution over classes)
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(output_data)

    # Map the index to the class label
    predicted_class_label = classes[predicted_class_index]

    return predicted_class_label


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result
