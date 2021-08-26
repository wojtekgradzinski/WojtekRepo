import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
from tqdm import tqdm


# Address
address_class = "css-e81eai"


df['address'] = "0"

address = business_pages[0].findAll('p', class_ = address_class)
address = address[1].text

df.loc[index, 'address'] = address
df.to_csv('yelp_dataset.csv', index = False)