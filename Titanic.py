#76% accuracry

# In[1]:

import pandas as pd
from pandas import DataFrame
import numpy as np

# In[2]:

train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")

# Preview the data
train_df.head()
train_df.info()

# In[3]:

#Choosing features from the training set

Y_train=train_df.iloc[:,1]
X_train=train_df.iloc[:,[2,4,5,6,7]]

X_test=test_df.iloc[:,[1,3,4,5,6]]

H=[X_train, X_test]
for change in H:
    change['Sex']=change['Sex'].map({'female':0,'male':1}).astype(int)

X_train=X_train.fillna(0) #Filling NAN values
X_test=X_test.fillna(0)

# In[4]:

from sklearn.neighbors import KNeighborsClassifier
###################################################################
#from sklearn import linear_model

#logreg = linear_model.LogisticRegression(C=100)

# we create an instance of Neighbours Classifier and fit the data.
#logreg.fit(X_train, Y_train)
#pred=logreg.predict(X_train)
###################################################################


knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

Y_predict=knn.predict(X_train)

# In[5]:

from sklearn.metrics import accuracy_score

accuracy=accuracy_score (Y_train, Y_predict)
print("Accuracy is {0:.2f}%".format(accuracy*100))

# In[6]:

result=knn.predict(X_test)

# In[7]:

index=test_df["PassengerId"]
Final_Output=pd.DataFrame(data=result,index=test_df["PassengerId"],columns=['Survived'])
Final_Output.to_csv(path_or_buf="../input/gender_submission.csv",header=True)
print('gender_submission.csv')













