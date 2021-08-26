import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
from tqdm import tqdm


attributes_outer_div = "arrange__373c0__UHqhV gutter-2__373c0__3Zpeq layout-wrap__373c0__34d4b layout-2-units__373c0__3CiAk border-color--default__373c0__2oFDT"

for index in tqdm(range(0,8)):

    df['url'][index]
    response = requests.get(df['url'][index])
    soup = BeautifulSoup(response.content, 'html.parser')

    attributes_elements = soup.findAll('div', class_ = attributes_outer_div)

    attributes = []
    for div_element in attributes_elements:

        for span_element in div_element.findAll('span')[1::2]:
            attributes.append(span_element.text)
    attributes = ",".join(attributes)