
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[14]:


df=pd.read_csv('data.csv')
df.head()


# In[50]:


df.fillna(df.mean())


# In[59]:


df.drop('Unnamed: 32',axis=1,inplace=True)


# In[60]:


pd.get_dummies(df['diagnosis'],drop_first=True)


# In[69]:


from sklearn.cross_validation import train_test_split
X = df.drop('diagnosis',axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[70]:


from sklearn.ensemble import RandomForestClassifier


# In[71]:


rfc = RandomForestClassifier(n_estimators=600)


# In[72]:


rfc.fit(X_train,y_train)


# In[73]:


predictions = rfc.predict(X_test)


# In[74]:


from sklearn.metrics import classification_report,confusion_matrix


# In[75]:


print(classification_report(y_test,predictions))


# In[68]:


print(confusion_matrix(y_test,predictions))


# In[ ]:




