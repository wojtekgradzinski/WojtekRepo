import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
from tqdm import tqdm

# Number of photos
# photos_class = "display--inline__373c0__1DbOG margin-l2__373c0__wvUpT border-color--default__373c0__2oFDT"
photos_class = "css-ardur"
regex = re.compile('[^0-9]')

df['photos_count'] = 0

num_photos = business_pages[0].findAll('span', class_ = photos_class)[4]
num_photos = int(regex.sub('', num_photos.text))

df.loc[index, 'photos_count'] = num_photos

df.to_csv('yelp_dataset.csv', mode = 'a', index = False)