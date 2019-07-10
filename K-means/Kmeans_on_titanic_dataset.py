import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('titanic.xls')
df.drop(['body','name'],1,inplace=True)
df.fillna(0,inplace=True)

#df.convert_objects(convert_numeric=True)
##
colums = df.columns.values
col=[]

##for getting colums with data type not int64 or float64
for i in colums:
    if df[i].dtype != np.int64 and df[i].dtype != np.float64:
        col.append(i)


for i in col:
    df[i] = df[i].astype('category')
    df[i] = df[i].cat.codes
    
df.drop(['sex'],1,inplace=True)
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)
labels = clf.labels_
correct= 0

for i in range(len(X)):
    if (labels[i]==y[i]):
        correct+=1
        
##for i in range(len(X)):
##    predict_me = np.array(X[i].astype(float))
##    predict_me = predict_me.reshape(-1,len(predict_me))
##    prediction = clf.predict(predict_me)
##    if prediction[0] == y[i]:
##        correct += 1
        
print(correct/len(X))
