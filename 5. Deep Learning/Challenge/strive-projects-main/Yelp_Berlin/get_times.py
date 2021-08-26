import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
from tqdm import tqdm

df['hours'] = "0"

hours_table = business_pages[0].findAll('table')[0]

rows = hours_table.findChildren('tr')

times = []
for idx, row in enumerate(rows):

    times_text = row.findAll('p', class_ = time_class)
    if times_text != []:
        times.append(row.findAll('p', class_ = time_class)[0].text)

days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
hours = {day:time for day in days for time in times}