# Grade: 0.16201  which is the error 

# In[1]:

import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[2]:

train=pd.read_csv("../input/leaf classification/train.csv")
test =pd.read_csv("../input/leaf classification/test.csv")


# In[3]:

train.head()

# In[4]:

Y_train=train.iloc[:,1]
X_train=train.iloc[:,2:]

X_test=test.iloc[:,1:]

# In[5]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1000000)
logreg.fit(X_train, Y_train)
pred=logreg.predict(X_train)

knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

Y_predict=knn.predict(X_train)

clf = KNeighborsClassifier(3)
clf.fit(X_train, Y_train)
test_predictions = clf.predict_proba(X_test)


# In[6]:

from sklearn.metrics import accuracy_score

accuracy=accuracy_score (Y_train, Y_predict)
accuracyy=accuracy_score (Y_train, pred)
print("KN Accuracy is {0:.2f}%".format(accuracy*100))
print("Logistic Accuracy is {0:.2f}%".format(accuracyy*100))

# In[7]:

output=list(np.unique(Y_predict))
X_test.shape

# In[8]:

submission = pd.DataFrame(test_predictions, columns=output)
submission.insert(0, 'id', test["id"])

submission.to_csv(path_or_buf="../input/species_submission.csv",header=True)
