#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import needed packages
# You may add or remove packages should you need them
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

# Set random seed
np.random.seed(0)

# Display plots inline and change plot resolution to retina
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# Set Seaborn aesthetic parameters to defaults
sns.set()


# In[4]:


# Load the Iris dataset included with scikit-learn
iris = load_iris()


# In[7]:


#iris


# In[11]:


df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris.head(5)


# In[119]:


df_iris["target"] = iris.target
df_iris["class"] = iris.target_names[iris.target]
df_iris.columns = [col.replace("(cm)", "").strip() for col in df_iris.columns]
df_iris


# In[15]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X = df_iris.iloc[:, 0:4]
scaler = StandardScaler()
# transform data
X_scaled = scaler.fit_transform(X)


# In[16]:





# In[70]:


pca = PCA()
pca


# In[75]:


data = pd.DataFrame(pca.fit_transform(X))
data


# In[82]:


# Put data in a pandas DataFrame
df = pd.DataFrame(data.values, columns=iris.feature_names)
df


# In[79]:


print(data)


# In[113]:


# Add target and class to DataFrame
df['target'] = iris.target
df['class'] = iris.target_names[iris.target]
df.columns = [col.replace("(cm)", "").strip() for col in df.columns]
df


# In[81]:


# Show 10 random samples
df.sample(10)


# # PCA Exercise
# Taking in consideration the iris dataset, answer the following questions. **You may have to run some code first :)**
# 1. How many **principal components can we consider**?
# 2. How do you think is going to be the **cumulated percentage of explained variance** attending to the number of components? Calculate it.
# 3. Consider the necessary number of components to explain at least a **99% of the variance**. Give the equations to calculate these components.
# 4. Calculate the **new values** for this decomposition and plot them.
# 5. Repeat the steps 3 and 4 **taking a 95% of the variance**

# In[33]:


# A graph to help you out
sns.set(style="ticks")
sns.pairplot(data = df_iris.loc[:,:"target"], hue = 'target')


# In[56]:


# Run the PCA model here
iris_pca = PCA().fit(X)
plt.plot(np.cumsum(iris_pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[65]:


pca_df = pd.DataFrame(pca.transform(X))
# You should end up with a transformed dataframe
pca_df


# In[58]:


# check the variance in each component
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)


# In[120]:


df_iris.head()


# In[127]:


# check the variance in each component
df_iris = df_iris.iloc[:,:4]
df_iris
pca_four_comp = PCA().fit(df_iris)

print(pca_four_comp.explained_variance_ratio_)


# In[128]:


# compare it with the original dataframe and to what it corresponds (more or less)
print(df.head(5))
print(pca_df.head(5))


# In[15]:


# plot it!


# In[138]:


chart = pca_four_comp.explained_variance_ratio_
plt.bar(range(0,4),height=chart, align="center")


# In[135]:


chart


# In[ ]:




