#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:42:38 2021

@author: mateo
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

df = pd.read_csv('yelp_dataset.csv')
print(df.columns)



#Highest Stars
highest_stars = df.sort_values(by = 'stars', ascending = False)
print(highest_stars['stars'], highest_stars['Restaurants_Name'])

#Highest reviews
highest_reviews = df.sort_values(by ='review_count', ascending = False)
print(highest_reviews['review_count'], highest_reviews['Restaurants_Name'])

#Correlation between highest reviews and highest stars            !!!!!insert corr into barplot
sr = highest_reviews['review_count'].corr(highest_stars['stars'])
print(sr)

#sns.barplot(data = df, x = 'stars', y = 'review_count')









#for n in range(len(df)):
#    if df.loc[n, 'review_count'] > np.average(df['review_count']) * 0.8:
#        rdf.assign(review_count = df.loc[n, 'review_count'] )




#Name Length 
g=1
length1 = []
for z in range(len(df)):
             try:
                 n = len(df.loc[z, 'Restaurants_Name'])
                 length1.append(n)
             except:
                 length1.append(None)
             
df['name_length'] = length1
             
#Name Length vs Stars

sns.lineplot(data = df, x = 'name_length', y = 'stars')

df['name_length'].corr(df['stars'])


#Number of reviews by category

a = df['categories'].values.tolist()
b = [x for x in a if str(x) != 'nan']
c = []
#for d in b:
#    c += b.split(",")
print(c)

all_cat = pd.DataFrame(data = c)
all_cat['Reviews'] = 0
h = len(all_cat)
print(all_cat)
#a = df[n, 'categories']
#for n in 
print(len(all_cat))

#for n in range(len(df)):
#   for a in range(h):
       
#print(b)






        
        
    
    
    
    