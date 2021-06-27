import pandas as pd
from sklearn import naive_bayes
from sklearn.feature_extraction import text
import numpy as np
import os 
import io 

def readFiles(path):
    for root, dirs, files in os.walk(path):
        for filename in files:
            path = os.path.join(root, filename)
            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if(inBody): lines.append(line)
                elif (line == '\n'): inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def createDataFrame(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return pd.DataFrame(rows, index=index)

data = pd.DataFrame({'message': [], 'class': []})
data = data.append(createDataFrame('./emails/spam', 'spam'))
data = data.append(createDataFrame('./emails/notspam', 'notspam'))

vectorizer = text.CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = naive_bayes.MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets) 

examples = ['Hey! Whats up?', 'Deal! Deal! Free deal!', 'Get cancer free now! Offers available', 'Dear Friend, Business Opportunity!']
examples = vectorizer.transform(examples)
print(classifier.predict(examples))
