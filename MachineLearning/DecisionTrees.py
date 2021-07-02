import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from six import StringIO  
from IPython.display import Image  
import pydotplus
import os     

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

filepath = '../assets/PastHires.csv'
df = pd.read_csv(filepath, header=0)

#sklearn needs every value to be numerical to for decision trees to work
mapping1 = {
    'Y': 1,
    'N': 0
}

mapping2 = {
    'BS': 0,
    'MS': 1,
    'PhD': 2 
}

df['Hired'] = df['Hired'].map(mapping1)
df['Employed?'] = df['Employed?'].map(mapping1)
df['Top-tier school'] = df['Top-tier school'].map(mapping1)
df['Interned'] = df['Interned'].map(mapping1)
df['Level of Education'] = df['Level of Education'].map(mapping2)

features = df.columns[:6]

y = df['Hired']
X = df[features]

#create a classfier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
# graph.write_png('decision_tree.png')

#using random forest to prevent overfitting
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10) #10 trees in random forest
clf = clf.fit(X, y)

# let's do some predictions
test1 = [4, 1, 4, 0, 1, 1]
print(clf.predict([test1]))
test2 = [4, 1, 0, 2, 1, 0]
print(clf.predict([test2]))
test3 = [10, 1, 0, 1, 1, 0]
print(clf.predict([test3]))
test4 = [4, 0, 2, 0, 1, 0]
print(clf.predict([test4]))
test5 = [0, 0, 0, 0, 0, 0]
print(clf.predict([test5]))



