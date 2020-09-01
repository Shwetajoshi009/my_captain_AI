#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import scipy
import pandas
import numpy
import matplotlib
import sklearn

print(scipy.__version__)


# In[2]:


#dependencies
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


# In[3]:


#loading the iris flower dataset
data=pandas.read_csv('Iris.csv')
pandas.DataFrame(data)
data.head()


# In[4]:


data=data.iloc[:,1:]
data.shape


# In[5]:


#statistical summary
data.describe()


# In[6]:


#class distribution
data.groupby('Species').size()


# In[7]:


#visualizing of data
#univariate plots- box and whisker plots
data.plot(kind='box',   subplots=True,layout=(2,2),  sharex=False, sharey= False)


# In[8]:


#histograms
data.hist()


# In[9]:


#multivariate plots
#interaction between variables
scatter_matrix(data)


# In[10]:


#creating validation dataset
#splitting dataset
array=data.values
x=array[:,0:4]
y=array[:,4]
x_train, x_val, y_train, y_val=train_test_split(x,y, test_size=0.2, random_state=1)


# In[11]:


#logistic regression
#linear discriminant analysis
#knn
#classification and regression trees
#guassian naive bayes
#support vector machines


model=[]
model.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
model.append(('LDA', LinearDiscriminantAnalysis()))
model.append(('KNN', KNeighborsClassifier()))
model.append(('NB', GaussianNB()))
model.append(('SVM', SVC(gamma='auto')))


# In[12]:


#evaluate the models
results=[]
names=[]
for name, m in model:
    kfold=StratifiedKFold(n_splits=10, random_state=None)
    cv_results=cross_val_score(m, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[14]:


#compare models
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()


# In[16]:


#make preds in svm
model=SVC(gamma='auto')
model.fit(x_train, y_train)
pred=model.predict(x_val)


# In[17]:


#evaluating predictions
print(accuracy_score(y_val, pred))
print(confusion_matrix(y_val, pred))
print(classification_report(y_val, pred))


# In[ ]:


#end

