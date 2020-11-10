# Make MyOwn Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from numpy.lib.npyio import save
from tqdm import tqdm
import random
import pickle
from config import DataDir, Categories, Img_size, pkl_label, pkl_feature, pkl_path


# Show image in Pet Images
def test_img():
    for category in Categories:
        path = os.path.join(DataDir, category)  # path to cats or dogs
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            plt.imshow(img_array, cmap="gray")
            plt.show()
            break
        break

def create_training_data():
    training_data = []
    for category in Categories:
        path = os.path.join(DataDir, category)
        # "cat" "dog" to 0 1
        class_num = Categories.index(category)
        # Run all img in Path (tqdm : progress bar)
        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (Img_size, Img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    
    # Shuffle Data (If not shuffle then tranning will think all img is cat or dog)
    random.shuffle(training_data)
    return training_data


def get_feature_label(training_data):
    features = []  # features
    labels = []  # label

    for feature, label in training_data:
        features.append(feature)
        labels.append(label)
        # np.array((labels, label))

    features = np.array(features).reshape(-1, Img_size, Img_size, 1)  # gray scale and reshape
    # x = np.array(x).reshape(-1, Img_size, Img_size, 3)    # RGB
    return features, labels

def save_pickle(fname, data):
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    pickle_out = open(fname, "wb")
    data = np.array(data)
    pickle.dump(data, pickle_out)
    pickle_out.close()

if __name__ == '__main__':
    training_data = create_training_data()
    features, labels = get_feature_label(training_data)

    save_pickle(pkl_feature, features)
    save_pickle(pkl_label, labels)