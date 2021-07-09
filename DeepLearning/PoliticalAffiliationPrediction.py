import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

features = ['party','handicapped-infants', 'water-project-cost-sharing', 
            'adoption-of-the-budget-resolution', 'physician-fee-freeze',
            'el-salvador-aid', 'religious-groups-in-schools',
            'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
            'mx-missle', 'immigration', 'synfuels-corporation-cutback',
            'education-spending', 'superfund-right-to-sue', 'crime',
            'duty-free-exports', 'export-administration-act-south-africa']

voting_data = pd.read_csv('../assets/house-votes-84.data.txt', names=features, na_values=['?'])
# print(voting_data.head(10))
# print(voting_data.describe())


voting_data.dropna(inplace=True)
# print(voting_data.head(10))
# print(voting_data.describe())

voting_data.replace(('y', 'n'), (1, 0), inplace=True)
voting_data.replace(('democrat', 'republican'), (1, 0), inplace=True)

# print(voting_data.head(10))

all_features = voting_data[features[1:]].values
target_classes = voting_data['party'].values

# print(all_features)
# print(target_classes)

from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

def getModel():
    model = Sequential()
    model.add(Dense(64, input_dim=16, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    return model

   
estimator = KerasClassifier(build_fn=getModel, epochs=100, verbose=0)
score = cross_val_score(estimator, all_features, target_classes, cv=10)
print(score.mean()) # 95.69 accuracy%
