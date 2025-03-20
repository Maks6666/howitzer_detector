from array import array

import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

model = KNeighborsClassifier()
model = joblib.load("/Users/maxkucher/opencv/howitzer_detector/analyzer/decision_maker.pkl")
params = model.get_params()
print(f"Analyzer model loaded: {params}")

