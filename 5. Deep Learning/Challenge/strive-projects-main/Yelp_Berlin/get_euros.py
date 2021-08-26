import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re

response = requests.get("https://www.yelp.co.uk/biz/maria-bonita-berlin-3")
soup = BeautifulSoup(response.content, 'html.parser')

# Euros
euros_class = "display--inline__373c0__1DbOG margin-r1__373c0__zyKmV border-color--default__373c0__2oFDT"

euros_category = soup.findAll('span', class_ = euros_class)
euroes_category = euros_category[1].text.strip()
# euros_category = euros_category[0].text.strip()

print(euros_category[1].text.strip())