import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
from tqdm import tqdm

photos_class = "css-ardur"
stars_class = "i-stars__373c0__1T6rz i-stars--large-4-half__373c0__2lYkD border-color--default__373c0__30oMI overflow--hidden__373c0__2B0kz"
reviews_class =  "arrange-unit__373c0__1piwO arrange-unit-fill__373c0__17z0h border-color--default__373c0__2oFDT nowrap__373c0__1_N1j"
claim_class = "border-color--default__373c0__2oFDT nowrap__373c0__1_N1j"
day_class = "day-of-the-week__373c0__124RF css-1h1j0y3"
time_class = "no-wrap__373c0__2vNX7 css-1h1j0y3"
stars_class = "display--inline__373c0__2SfH_ border-color--default__373c0__30oMI"
categories_class = "css-bq71j2" 
attributes_outer_div = "arrange__373c0__UHqhV gutter-2__373c0__3Zpeq layout-wrap__373c0__34d4b layout-2-units__373c0__3CiAk border-color--default__373c0__2oFDT"

df = pd.read_csv('yelp_dataset.csv')
days = ['Monday','Tuesday','Wednesday','Thursday','Friday']

total_webpages = range(160,241)
n = 8
sessions = [total_webpages[i:i+n] for i in range(0, len(total_webpages), n)]

# print(sessions)
for session, session_range in tqdm(enumerate(sessions)):
    # print(session)
    for index in tqdm(session_range):

        # # with open(f"./webpages/business_page{index}.html", encoding = "utf8") as f:
        # #     soup = BeautifulSoup(f, 'html.parser')
        df['url'][index]
        response = requests.get(df['url'][index])
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Numbre of photos
        try:
            regex = re.compile('[^0-9]')

            num_photos = soup.findAll('span', class_ = photos_class)[4]
            num_photos = int(regex.sub('', num_photos.text))

            df.loc[index, 'photos_count'] = num_photos
            
        
        except (ValueError, IndexError):
            df.loc[index, 'photos_count'] = None

        # Stars
        # ratings_class ="i-stars__373c0__1T6rz i-stars--large-4-half__373c0__2lYkD border-color--default__373c0__30oMI overflow--hidden__373c0__2B0kz" 
        # stars_class = "i-stars__373c0__1T6rz i-stars--large-4-half__373c0__2lYkD border-color--default__373c0__30oMI overflow--hidden__373c0__2B0kz"
        regex = re.compile('[^0-9.]')

        stars = soup.findAll('span', class_ = stars_class)[0]
        stars = stars.findChildren('div')
        stars = stars[0]['aria-label']
        stars = regex.sub('', stars)

        df.loc[index, 'stars'] = stars

        # Review count

        regex = re.compile('[^0-9]')

        review_count = soup.findAll('div', class_ = reviews_class)
        review_count = review_count[0].text
        review_count = int(regex.sub('', review_count))

        df.loc[index, 'review_count'] = review_count

        # Claimed vs Unclaimed

        claimed = soup.findAll('div', class_ = claim_class)

        if claimed[0].text.strip() == "Claimed":
            claimed = 1
        else: 
            claimed = 0

        df.loc[index, 'claimed'] = claimed

        # Address

        address = soup.findAll('address', class_ = "")

        location = ""
        for element in address:

            for p in element.findAll('p')[:-1]:
                location += p.text

        df.loc[index, 'address'] = location

        # Hours

        hours_table = soup.findAll('table')[0]

        rows = hours_table.findChildren('tr')

        times = []
        for idx, row in enumerate(rows):

            times_text = row.findAll('p', class_ = time_class)
            if times_text != []:
                times.append(row.findAll('p', class_ = time_class)[0].text)

        
        hours = {day:time for day,time in zip(days,times)}

        df.loc[index, 'hours'] = str(hours)

        # Categories
        categories_elements = soup.findAll('span', class_ = categories_class)

        categories = []
        for element in categories_elements:

            for category in element.findAll('a'):
                categories.append(category.text)

        categories = ",".join(categories)

        df.loc[index,'categories'] = categories

        # Attributes

        attributes_elements = soup.findAll('div', class_ = attributes_outer_div)

        attributes = []
        for div_element in attributes_elements:

            for span_element in div_element.findAll('span')[1::2]:
                attributes.append(span_element.text)
        attributes = ",".join(attributes)

        df.loc[index,'attributes'] = attributes

        # Euros
        euros_class = "css-1xxismk"

        euros_category = soup.findAll('span', class_ = euros_class)
        euros_category = len(euros_category[0].text.strip())

        df.loc[index,'euros'] = euros_category
        
        # Saving to dataset
        df.to_csv('yelp_dataset.csv', index = False)
        delay = np.random.randint(40,100)
        time.sleep(delay)
    
    time.sleep(np.random.randint(150,1200))