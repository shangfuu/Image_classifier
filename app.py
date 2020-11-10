# Application 
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from config import Categories, test_path, model_name


model = tf.keras.models.load_model(model_name)

for img_name in os.listdir(test_path):
    # Predict Data
    Img_size = 100  # 50 in txt-based
    img_gray = cv2.imread(os.path.join(test_path, img_name), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        continue
    img_gray = cv2.resize(img_gray, (Img_size, Img_size))
    img_gray = img_gray.reshape(-1, Img_size, Img_size, 1)
    # prediction
    prediction = model.predict(img_gray)

    # show image
    img = cv2.imread(os.path.join(test_path, img_name))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    title = 'I Guess I Am a %s ' % Categories[int(prediction[0][0])]
    plt.imshow(img_RGB)
    plt.axis('off')
    plt.title(title)
    plt.show()

    cv2.putText(img, title, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(title, img)
    if not os.path.exists('result'):
        os.makedirs('result')
    cv2.imwrite(os.path.join('result', img_name), img)
    cv2.waitKey(0)
    
