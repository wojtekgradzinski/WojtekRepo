#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn import metrics    
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np 
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting 
# Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier
import joblib
import sys
sys.path.append(r"C:\Users\Wojtek\OneDrive\Desktop\STRIVE SCHOOL\WojtekRepo\GROUP PROJECT\ROCK SCISORS PAPER")


# In[7]:


#Rock= 0
#Paper=1
#Scissor=2


# In[8]:


path = 'images'
data1 = pd.DataFrame ()
data2= pd.DataFrame()
for folder in os.listdir(path):
    if folder == 'train':
    
        for f in os.listdir (path + '/' + folder):
            class_data = np.zeros ( (len(os.listdir (path + '/' + folder + '/' + f )), 1025) )
            print('Original shape')
            print (class_data.shape)
        
            for i, img_name in enumerate (os.listdir (path + '/' + folder + '/' + f )):

                img = Image.open (path + '/' + folder + '/' + f + '/' + img_name)
                img_arr = np.array (img, dtype = int)
                img_arr = img_arr.flatten()
                
                class_data [i, :1024] = img_arr
                class_data [i, 1024]  = int (f)  #assigning target to the last column 

            class_data = pd.DataFrame (class_data)
            data1 = pd.concat ([data1, class_data])
            print('Size after concatination')
            print(data1.shape)
    else:
        for f in os.listdir (path + '/' + folder):
            class_data = np.zeros ( (len(os.listdir (path + '/' + folder + '/' + f )), 1025) )
            print('Original shape')
            print (class_data.shape)
        
            for i, img_name in enumerate (os.listdir (path + '/' + folder + '/' + f )):

                img = Image.open (path + '/' + folder + '/' + f + '/' + img_name)
                img_arr = np.array (img, dtype = int)
                img_arr = img_arr.flatten()
                
                class_data [i, :1024] = img_arr
                class_data [i, 1024]  = int (f)  #assigning target to the last column 

            class_data = pd.DataFrame (class_data)
            data2 = pd.concat ([data2, class_data])
            print('Size after concatenation')
            print(data2.shape)

            
data1.to_csv ('train_game.csv')
data2.to_csv('test_game.csv')

            


# In[9]:


data2.iloc[:,-1].unique()
data2.shape


# In[10]:



#checking target column

data1.iloc[:,-1].unique()
data1.shape


# In[11]:


X_tr = data1.iloc[:,:1024]
y_tr = data1.iloc[:,1024]

X_val = data2.iloc[:,:1024]
y_val = data2.iloc[:,1024]


# In[12]:


scaler = StandardScaler()


data_norm_train = scaler.fit_transform(X_tr)
data_norm_test = scaler.transform(X_val)


# In[13]:


#sns.histplot(data=data_norm, bins=10)


# In[14]:


tree_classifiers = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":   ExtraTreesClassifier(),
  "Random Forest": RandomForestClassifier(),
  "AdaBoost":      AdaBoostClassifier(),
  "Skl GBM":       GradientBoostingClassifier(),
  "Skl HistGBM":   GradientBoostingClassifier(),
  "XGBoost":       XGBClassifier(),
  "LightGBM":      LGBMClassifier(),
  "CatBoost":      CatBoostClassifier() 
}


# In[53]:


x_train, x_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)




results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})


for model_name, model in tree_classifiers.items(): # FOR EVERY PIPELINE (PREPRO + MODEL) -> TRAIN WITH TRAIN DATA (x_train)
    start_time = time.time()
    model.fit(x_train, y_train)
    pred = model.predict(X_val)    # GET PREDICTIONS USING x_val
    total_time = time.time() - start_time

    results = results.append({"Model":    model_name,
                              "Confusion Matrix": confusion_matrix(y_val, pred),
                              "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
                              
                              

 
# Your code goes here


results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')


# In[54]:


#saving model

model = tree_classifiers["LightGBM"]
model_choice = 'best_model.sav'
joblib.dump(model, model_choice)


# In[55]:


#loading model
loaded_model = joblib.load(model_choice)
result = loaded_model.score(X_val, y_val)
print(result)


# In[65]:


import cv2
import matplotlib.pyplot as plt

rawimage = cv2.imread(r'C:\Users\Wojtek\OneDrive\Desktop\STRIVE SCHOOL\WojtekRepo\GROUP PROJECT\ROCK SCISORS PAPER\handgestures\hands\s2.jpg')

plt.imshow(rawimage, cmap= 'gray')


# In[66]:


#Transform the Image into 32 x 32 skeleton using transform_image.py

from handgestures.transform_image import transform_single_image

try:
    img = transform_single_image(rawimage)
    plt.imshow(img, cmap="gray")
except:
    print('Please try with another image!')



# In[ ]:




