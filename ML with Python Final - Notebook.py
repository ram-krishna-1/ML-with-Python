#!/usr/bin/env python
# coding: utf-8

# In this notebook we load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.


# Assignment Prepared by IBM Machine Learning with Python
# Assignment completed by Ram Krishna

# In[1]:

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |


# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 

# In[6]:


df['loan_status'].value_counts()


# In[7]:

#!conda install -c anaconda seaborn -y


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay their loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[17]:


X = Feature
X[0:5]


# What are our labels?

# In[18]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# Use the following algorithms:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 


# # K Nearest Neighbor(KNN)

# In[20]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

import scipy.optimize as opt
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import jaccard_score

Ks = 20
acc = np.zeros((Ks - 1))
for n in range(1, Ks):
    neighbors = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    y_hat_k = neighbors.predict(X_test)
    acc[n - 1] = jaccard_score(y_test, y_hat_k, pos_label = "PAIDOFF")

acc


# In[22]:


print("Best Accuracy:", acc.max(), "at k =", acc.argmax() + 1)

k = 7
neigh_7 = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
y_hat_k = neigh_7.predict(X_test)

print("Jaccard Score:", jaccard_score(y_test, y_hat_k, pos_label = "PAIDOFF"))
print("F1 Score:", f1_score(y_test, y_hat_k, average = "weighted"))


# # Decision Tree

# In[23]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = "entropy")
tree.fit(X_train, y_train)
predict_tree = tree.predict(X_test)
predict_tree[0:5], y_test[0:5]


# In[24]:


print("Jaccard Score:", jaccard_score(y_test, predict_tree, pos_label = "PAIDOFF"))
print("F1 Score:", f1_score(y_test, predict_tree, average = "weighted"))


# # Support Vector Machine

# In[25]:


from sklearn import svm
clf = svm.SVC(kernel = "rbf")
clf.fit(X_train, y_train)


# In[26]:


y_hat_svm = clf.predict(X_test)


# In[27]:


from sklearn.metrics import f1_score
print("F1 Score:", f1_score(y_test, y_hat_svm, average='weighted'))
print("Jaccard Score:", jaccard_score(y_test, y_hat_svm, pos_label = "PAIDOFF"))


# # Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C = 0.01, solver = "lbfgs").fit(X_train, y_train)


# In[29]:


y_hat_LR = LR.predict(X_test)
y_hat_LR_proba = LR.predict_proba(X_test)


# In[30]:


print("F1 Score:", f1_score(y_test, y_hat_LR, average = "weighted"))
print("Jaccard Score:", jaccard_score(y_test, y_hat_LR, pos_label = "PAIDOFF"))
print("LogLoss Score:", log_loss(y_test, y_hat_LR_proba))


# # Model Evaluation using Test set

# In[31]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# Download and load the test set:

# In[32]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[33]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[34]:


df = test_df

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['dayofweek'] = df['effective_date'].dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

df.groupby(['education'])['loan_status'].value_counts(normalize=True)

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)

X = Feature

y = df['loan_status'].values

X = preprocessing.StandardScaler().fit(X).transform(X)


# In[35]:


#KNN

yhat_KNN=neigh_7.predict(X)
KNN_Jaccard = jaccard_score(y, yhat_KNN, pos_label = "PAIDOFF")
KNN_F1 = f1_score(y, yhat_KNN, average='weighted')

#DecisionTree

yhat_tree = tree.predict(X)
Tree_Jaccard = jaccard_score(y, yhat_tree, pos_label = "PAIDOFF")
Tree_F1 = f1_score(y, yhat_tree, average = "weighted")

#SVM

yhat_SVM = clf.predict(X)
SVM_Jaccard = jaccard_score(y, yhat_SVM, pos_label = "PAIDOFF")
SVM_F1 = f1_score(y, yhat_SVM, average = "weighted")

#Logistic

yhat_logistic = LR.predict(X)
yhat_logistic_proba = LR.predict_proba(X)
logistic_Jaccard = jaccard_score(y, yhat_logistic, pos_label = "PAIDOFF")
logistic_F1 = f1_score(y, yhat_logistic, average = "weighted")
logistic_logloss = log_loss(y, yhat_logistic_proba)


# In[36]:


Algorithm = ["KNN", "Decision Tree", "SVM", "LogisticRegression"]
Jaccard = [KNN_Jaccard, Tree_Jaccard, SVM_Jaccard, logistic_Jaccard]
F1_Score = [KNN_F1, Tree_F1, SVM_F1, logistic_F1]
LogLoss = ["NA", "NA", "NA", logistic_logloss]

pd.DataFrame(data=
            {"Algorithm": Algorithm, "Jaccard": Jaccard, "F1-score": F1_Score, "LogLoss": LogLoss},
            columns = ["Algorithm", "Jaccard", "F1-score", "LogLoss"], index = None)



# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
