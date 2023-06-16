# Finding predictive pixels form intrincic optic imaging data 

The object of this project is to find pixels (activation maps recoreded via optic imaging when monkey passively observing different types of objects) that are predictive for different objects. 

[model_training.py](https://github.com/Anmin-Yang/optic_classifier/blob/main/model_training.py) acquires coeficients of each pixel via linear SVC (in OVO fashion).

[bootstrap.py](https://github.com/Anmin-Yang/optic_classifier/blob/main/bootstrap.py) determines predictive voxels via 1,000 times bootstrap (with uncorrected p-value at 0.05, subject to change).