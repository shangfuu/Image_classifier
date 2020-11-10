import os

root_path = os.path.abspath(os.path.abspath(os.getcwd()))
# prepare.py
DataDir = root_path + "/PetImages"
Categories = ["Dog", "Cat"]
Img_size = 100

pkl_path = "Data/pickles"
pkl_feature = os.path.join(pkl_path, "features.pickle")
pkl_label = os.path.join(pkl_path, "labels.pickle")

# train.py
model_path = 'Data/models'
model_name = os.path.join(model_path, '64x3-CNN.model')
log_name = 'Cats-vs-dogs-64x2-CNN'

# app.py
test_path = root_path + "/testData"
