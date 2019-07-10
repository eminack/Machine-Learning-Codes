import numpy as np
from sklearn.model_selection import cross_validate , train_test_split
from sklearn import preprocessing ,neighbors
import pandas as pd
import pickle

accuracies = []
for l in range(25):
    df=pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)

    X = np.array(df.drop(['class'],1))
    y = np.array(df['class'])

    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

    clf=neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train,y_train)

    #with open('kneighbors.pickle','wb') as f:
    #   pickle.dump(clf,f)

    #pickle_in = open('kneighbors.pickle','rb')
    #clf = pickle.load(pickle_in)

    accuracy = clf.score(X_test,y_test)
    print(accuracy)

    #prediction
    #eg_measure = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
    #eg_measure = eg_measure.reshape(len(eg_measure),-1)
    #prediction = clf.predict(eg_measure)
    #print(prediction)
    accuracies.append(accuracy)

print('ans = ' + str(sum(accuracies)/len(accuracies)))
