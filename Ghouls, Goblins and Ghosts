# Grade : 0.72967  

# In[1]:

import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# In[2]:

#Reading the datasets
train= pd.read_csv('../input/Ghouls Goblins and Ghosts/train.csv')
test = pd.read_csv('../input/Ghouls Goblins and Ghosts/test.csv')

train.head()

# In[3]:

print(np.sort(train['color'].unique()))
print(np.sort(test['color'].unique()))
print(np.sort(train['type'].unique()))

# In[4]:

#Filling NA values
train=train.fillna(0) 
test=test.fillna(0)
#Setting values for Color
H=[train, test]
for change in H:
    change['color']=change['color'].map({'clear':0,'green':1,'black':2,'white':3,'blue':4,'blood':5}).astype(float)
    
train.head()

# In[5]:

test.head()


# In[6]:

#Extracting Data and Labels

Y_train=train.iloc[:,6]
X_train=train.iloc[:,[1,2,3,4,5]]

X_test=test.iloc[:,[1,2,3,4,5]]


# In[7]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1000000)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)
pred=logreg.predict(X_train)

knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

Y_predict=knn.predict(X_train)

# In[8]:

from sklearn.metrics import accuracy_score

accuracy=accuracy_score (Y_train, Y_predict)
accuracyy=accuracy_score (Y_train, pred)
print("Accuracy is {0:.2f}%".format(accuracy*100))
print("Accuracyy is {0:.2f}%".format(accuracyy*100))

# In[9]:

result=logreg.predict(X_test)

# In[10]:

index=test["id"]
Final_Output=pd.DataFrame(data=result,index=test["id"],columns=['type'])
Final_Output.to_csv(path_or_buf="../input/type_submission.csv",header=True)
print('type_submission.csv')
