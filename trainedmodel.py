import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


#split test and training data sets
def data_split(data, ratio):
    np.random.seed(42)
    shuffeled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffeled[:test_set_size]
    train_indices = shuffeled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__=="__main__":
    sympdata = pd.read_csv('coviddata.csv')
    train, test = data_split(sympdata, 0.2)

    trainArray1 = train[['fever','pain','age','noseRunning','breatingDifficulty']].to_numpy()
    testArray1 = test[['fever','pain','age','noseRunning','breatingDifficulty']].to_numpy()

    trainArray2 = train[['probabilityInfection']].to_numpy().reshape(480 ,)
    testArray2 = test[['probabilityInfection']].to_numpy().reshape(119 ,)

    #train data sets with LogisticRegression
    clf = LogisticRegression()
    clf.fit(trainArray1, trainArray2)
    
    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()
    

    #inference code
    sampleInput1 = [100.3223, 1, 23, 0, 1]

    sampleOutputPrediction1 = clf.predict([sampleInput1]) #to get the prediction
    print (sampleOutputPrediction1)

    sampleOutputProbability1 = clf.predict_proba([sampleInput1])[0][1] #to get the probability
    print (sampleOutputProbability1)