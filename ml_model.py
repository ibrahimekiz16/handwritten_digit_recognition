import numpy as np
from tensorflow.keras.models import load_model
from cv2 import resize

model = load_model("mnist_model.h5")

def predict_digit(img_array):
    """
    img_array: 280x280 canvas'tan gelen numpy array
    """
    
    img_resized = resize(img_array[0,:,:,0], (28,28))
    img_resized = img_resized.reshape(1,28,28,1)
    prediction = model.predict(img_resized)
    digit = np.argmax(prediction)
    return digit
