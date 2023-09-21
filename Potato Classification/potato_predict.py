import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from io import BytesIO


def predict(image_url):

    # Load the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Preprocess the image to match model's input dimensions
    image = image.resize((224, 224))  # Adjust the dimensions
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Load the trained model
    model = keras.models.load_model('potato_model.keras')

    # Make predictions
    predictions = model.predict(image)

    # Process the predictions (e.g., get class labels)
    class_labels = ["Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight"] 
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_labels[predicted_class[0]]

    return predicted_label

if __name__ == "__main__":
    print("Call predict function with image-url as parameter")
    # print(predict("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTsWENHLCbDHxJW9NZZfbNctX5LXqonLbUPlA&usqp=CAU"))