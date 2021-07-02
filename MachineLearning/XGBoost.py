from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
iris = load_iris()

numSamples, numFeatures = iris.data.shape


Xtrain, Xtest, ytrain, ytest = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

trainingdata = xgb.DMatrix(Xtrain, label=ytrain)
testingdata = xgb.DMatrix(Xtest, label=ytest)

params = {
    'max_depth': 3,
    'eta': 0.29,
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss'
}

epochs = 10

model = xgb.train(params, trainingdata, epochs)

predictions = model.predict(testingdata)

print(predictions)

accuracy = accuracy_score(ytest, predictions)

print('accuracy: ', accuracy)
