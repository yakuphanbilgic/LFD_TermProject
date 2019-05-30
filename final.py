from numpy import *
from sklearn import svm
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

def loadData(trainPath, testPath):
    #reading the train data
    #it has 0...595 columns, all features except last one is label
    trainData = genfromtxt(trainPath, delimiter= ',', skip_header = 1, dtype = None, usecols = [*range(0, 595)])

    #reading the label here
    labels = genfromtxt(trainPath, delimiter=',', skip_header=1, dtype=None, usecols = (595))

    #reading test data
    testData = genfromtxt(testPath, delimiter = ',', skip_header = 1, dtype = None)

    return trainData, labels, testData
    
def preprocessing(trainData, testData):
    trainData = trainData.copy()
    trainData = trainData.view((float, len(trainData.dtype.names)))
    maxAbsTransformer = RobustScaler(quantile_range=(10, 90)).fit(trainData)
    trainDataScaled = maxAbsTransformer.transform(trainData)

    testData = testData.copy()
    testData = testData.view((float, len(testData.dtype.names)))
    maxAbsTransformer = RobustScaler(quantile_range=(10, 90)).fit(testData)
    testDataScaled = maxAbsTransformer.transform(testData)
    
    pca = PCA().fit(trainDataScaled)
    plt.plot(cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    
    pca = PCA().fit(testDataScaled)
    plt.plot(cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    
    pca = PCA(n_components = 70, whiten=True, random_state = 1)
    pca.fit(trainDataScaled)
    trainDataT = pca.transform(trainDataScaled)
    
    pca = PCA(n_components = 70, whiten=True, random_state = 1)
    pca.fit(testDataScaled)
    testDataT = pca.transform(testDataScaled)
    
    return trainDataT, testDataT
    
def trainModel(trainData, labels):
    clf = svm.SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0,
        decision_function_shape='ovr', degree=2, gamma= 'auto',
        kernel='rbf', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)
    clf.fit(trainData, labels)  

    return clf

def predict(model, testData):
    output = empty((len(testData),2), dtype=int)
    
    for i in range(len(testData)):
        oneSample = reshape(testData[i], (1,-1))
        output[i][0] = i + 1
        output[i][1] = model.predict(oneSample)
    
    return output

def writeOutput(output):
    savetxt("submission.csv", output, header="ID,Predicted", fmt='%i',delimiter=",", comments='')
    
def findParameters(trainDataT, labels):
    #Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 'auto'],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]},
                   {'kernel': ['linear'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 'auto'],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]}
                   ]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(trainDataT, labels)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
    
def main():
    trainData, labels, testData = loadData("/Users/yakuphanbilgic/train.csv", "/Users/yakuphanbilgic/test.csv")
    
    trainDataT, testDataT = preprocessing(trainData, testData)
    
    svmModel = trainModel(trainDataT, labels)
    
    findParameters(trainDataT, labels)
    
    output = predict(svmModel, testDataT)

    writeOutput(output)

if __name__ == "__main__":
    main()
    
    
