import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
from tqdm import tqdm

# Review count
reviews_class =  "css-bq71j2"

regex = re.compile('[^0-9]')

df['review_count'] = 0

review_count = business_pages[0].findAll('span', class_ = reviews_class)
review_count = review_count[0].text
review_count = int(regex.sub('', review_count))

df.loc[index, 'review_count'] = review_count
df.to_csv('yelp_dataset.csv', index = False)