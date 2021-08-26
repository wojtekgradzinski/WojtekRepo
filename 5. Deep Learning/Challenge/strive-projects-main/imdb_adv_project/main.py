#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bs4')


# In[ ]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


page = requests.get("https://www.imdb.com/list/ls091520106/?st_dt=&mode=detail&page=1&sort=user_rating,desc&ref_=ttls_ref_gnr&genres=Adventure")

soup = BeautifulSoup(page.content, 'html.parser')

all_data = soup.find_all('div', class_ = "lister-item-content")


# Lists for data Storage
movie_name = []
desc = []
release_date = []
director = []
rating = []
duration = []
genre = []
actors = []
filming_dates = []
gross = []


######## Fill lists with extracted data ########


for data in all_data:
    movie_name.append(data.find('a').get_text(separator=' '))
    desc.append(data.find('p', class_= "").get_text(separator=' ')[5:])
    release_date.append(int(data.find('span', class_="lister-item-year text-muted unbold").get_text(separator=' ').strip('()I ')))
    rating.append(data.find('span', class_ = 'ipl-rating-star__rating').get_text(separator=' '))
    duration.append(int(data.find('span', class_ = 'runtime').get_text(separator=' ')[:-4]))
    genre.append(data.find('span', class_ = 'genre').get_text(separator=' ')[1:-12].split(', '))



people = soup.find_all('p', class_ ='text-muted text-small')

for person in people[1::3]:
    person1= person.get_text(separator=' ')
    splittedPerson1 = person1.split('|')
    splittedPerson1a = splittedPerson1[0][16:]
    splittedPerson1b = splittedPerson1[1][15:]
    final_actors = splittedPerson1b.replace(' \n', '')
    final_director = splittedPerson1a.replace(' \n', '')
    director.append(final_director[:-1].split(' , '))
    actors.append(final_actors.split(' , '))
    


values = soup.find_all('p', class_ = 'text-muted text-small')

for value in values[2::3]:
    raw_text = value.get_text(separator=' ')
    values_splitted = raw_text.split('|')
    gross_v = values_splitted[1].strip('Gross: ')
    gross_f = gross_v.replace('$', '').replace('M', '')
    gross.append(float(gross_f.replace('\n ', '').replace(' \n', '')))


######## Create Pandas DataFrame ########

dataList = zip(movie_name, genre, desc, release_date, director, actors, rating, duration, gross)
movieData = pd.DataFrame(dataList, columns = ['Title', 'Genre', 'Summary', 'Release Date', 'Directors', 'Actors', 'Ratings', 'Duration', 'Gross Income'])

print(movieData)


# In[ ]:


movieData.head()


# ### graphs

# In[ ]:


# libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns


# In[ ]:


#RATING + DURATION
#CONCLUSION : IN ADVENTURE, LONGER MOVIES WERE SLIGHTLY MORE APPRECIATED


plt.figure(figsize=(10, 10))
plt.scatter(
    x = movieData['Ratings'], 
    y = movieData['Duration'], 
    s=2000,
    c="magenta", 
    alpha=0.6, 
    edgecolors="white", 
    linewidth=2);

plt.xlabel('Ratings')
plt.ylabel('Duration')


# In[ ]:


#RATING + DIRECTOR

plt.figure(figsize=(10, 10))
movieData.groupby('Directors')['Ratings'].nunique().plot(kind='bar', cmap='RdYlBu')
plt.show()


# In[ ]:


#RELEASE YEAR + BOX OFFICE

plt.figure(figsize=(10, 10))
sns.scatterplot(data=movieData, x="Release Date", y="Gross Income", legend=False, s = 2000, cmap="Accent", alpha=0.7, edgecolors="grey", linewidth=2)


# In[ ]:





# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=21791e50-29ae-4ca2-93bf-1753be04b754' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
