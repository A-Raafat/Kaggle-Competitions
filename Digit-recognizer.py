# Grade: 0.90771 

# In[1]:

import numpy as np
import pandas as pd
from pandas import DataFrame

# In[2]:

train=pd.read_csv("../input/digit recognizer/train.csv")
test =pd.read_csv("../input/digit recognizer/test.csv")

# In[3]:

Y_train=train.iloc[:,0]
X_train=train.iloc[:,1:]

X_test=test

# In[4]:

from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1000000)
logreg.fit(X_train, Y_train)
pred=logreg.predict(X_train)

# In[5]:

Result=logreg.predict(X_test)

# In[6]:

xx=np.arange(Result.shape[0])
Result=Result.astype("int")

# In[7]:
op=pd.DataFrame(data=Result)

# In[8]:
op.to_csv(path_or_buf="../input/digit_submission.csv",header=True)

