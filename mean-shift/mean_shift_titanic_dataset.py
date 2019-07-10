import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('titanic.xls')
original_df=pd.DataFrame.copy(df)
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
    
#df.drop(['sex'],1,inplace=True)
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)
labels = clf.labels_
cluster_centers = clf.cluster_centers_
n_clusters_ = len(np.unique(labels))
original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived']==1) ]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate
    
print(survival_rates)
    
##for i in range(len(X)):
##    if (labels[i]==y[i]):
##        correct+=1
        
##for i in range(len(X)):
##    predict_me = np.array(X[i].astype(float))
##    predict_me = predict_me.reshape(-1,len(predict_me))
##    prediction = clf.predict(predict_me)
##    if prediction[0] == y[i]:
##        correct += 1
        
#print(correct/len(X))
