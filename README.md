# Image_classifier
Classify photos of Dogs and Cats.

> Dataset: https://www.microsoft.com/en-us/download/details.aspx?id=54765
## Data
* Organise the dataset as follows:
```
Data
    pickles
        features.pickle
        labels.pickle
    models
        64x3-CNN.model
testData
    (images)    
PetImages
    Cat
        (images)
    Dog
        (images)
```
## Install
* Type <code>pip install -r requirements.txt</code> to install all packages.
* Please check config.py for following steps.

## Prepare dataset
Type <code> python prepare.py</code> to make pickle files for your PetImages.
## Training
Type <code>python train.py </code> to train simple model/
Type <code>python train.py -m hard</code> to train model with more accuracy one.
## Testing
Type <code>python app.py</code> to test your cat / dog images in testData folder.

<img src="/result/cat1.jpeg">
<img src="/result/dog1.jpg">