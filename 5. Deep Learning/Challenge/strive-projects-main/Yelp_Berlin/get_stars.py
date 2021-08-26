import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re

# Stars
# ratings_class ="i-stars__373c0__1T6rz i-stars--large-4-half__373c0__2lYkD border-color--default__373c0__30oMI overflow--hidden__373c0__2B0kz"
stars_class = "i-stars__373c0__1T6rz i-stars--large-4-half__373c0__2lYkD border-color--default__373c0__30oMI overflow--hidden__373c0__2B0kz"
regex = re.compile('[^0-9.]')

df['stars'] = "0"

stars = business_pages[0].findAll('div', class_ = stars_class)
stars = stars[0]['aria-label']
stars = regex.sub('', stars)

df.loc[index, 'stars'] = stars
df.to_csv('yelp_dataset.csv', index = False)