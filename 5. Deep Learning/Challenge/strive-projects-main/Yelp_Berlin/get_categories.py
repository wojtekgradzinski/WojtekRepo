import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
from tqdm import tqdm

categories_class = "css-bq71j2" 

for index in tqdm(range(0,8)):

    df['url'][index]
    response = requests.get(df['url'][index])
    soup = BeautifulSoup(response.content, 'html.parser')

    categories_elements = soup.findAll('span', class_ = categories_class)

    categories = []
    for element in categories_elements:
        category.findAll('a')

        for category in element.findAll('a'):
            categories.append(category.text)

    categories = ",".join(categories)