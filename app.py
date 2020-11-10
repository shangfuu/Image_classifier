# Application 
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from config import Categories, test_path, model_name


model = tf.keras.models.load_model(model_name)

for img in os.listdir(test_path):
    # Predict Data
    Img_size = 100  # 50 in txt-based
    img_array = cv2.imread(os.path.join(test_path, img), cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        continue
    new_array = cv2.resize(img_array, (Img_size, Img_size))
    new_array = new_array.reshape(-1, Img_size, Img_size, 1)
    # prediction
    prediction = model.predict(new_array)

    # show image
    img = cv2.imread(os.path.join(test_path, img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    # Not showing Coordinate size
    plt.axis('off')
    plt.title('I Guess I Am a %s ' % Categories[int(prediction[0][0])])
    plt.show()

