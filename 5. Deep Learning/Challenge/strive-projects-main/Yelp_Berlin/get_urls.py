import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
import re
from tqdm import tqdm

# Business urls
# https://www.yelp.co.uk/search?cflt=homeservices&find_loc=Berlin%2C%20Germany
# https://www.yelp.co.uk/search?cflt=restaurants&find_loc=Berlin%2C%20Germany

def get_num_pages(business):

    num_page_class = "css-e81eai"

    response = requests.get(f"https://www.yelp.co.uk/search?cflt={business}&find_loc=Berlin%2C%20Germany")
    soup = BeautifulSoup(response.content,'html.parser')

    num_pages = soup.findAll('span', class_ = num_page_class)
    num_pages = int(num_pages[-3].text[-2:])

    return num_pages



# other pages url (for restaurants case)
# https://www.yelp.co.uk/search?cflt=restaurants&find_loc=Berlin%2C%20Germany&start=10
# pages --> main_url + (start = range(10,231,10)), can automate finding ending
# index

def get_urls(business):
    num_pages = get_num_pages(business)

    main_url = f"https://www.yelp.co.uk/search?cflt={business}&find_loc=Berlin%2C%20Germany"
    responses = [requests.get(main_url)]

    for index in tqdm(range(10, num_pages*10, 10), desc=f"Requesting pages for {business}"):
        responses.append(requests.get(f"{main_url}&start={index}"))
        delay = np.random.randint(1,6)
        time.sleep(delay)

    webpages = []
    for response in responses:
        webpages.append(BeautifulSoup(response.content, 'html.parser'))

    # Class to get hrefs
    span_class = "css-1pxmz4g"

    # Needed regex here as strip was leaving some characters Strangely those
    # characters only appeared on appending but not in a direct print

    regex = re.compile('[^a-zA-Z]')
    child_urls = []
    names = []
    for webpage in webpages:
        items = webpage.findAll('span', class_ = span_class)
        
        for item in items:

            if item.find('a') != None:
                names.append(regex.sub('', item.text))
                child_urls.append(f"https://www.yelp.co.uk{item.find('a')['href']}")

    df = pd.DataFrame({"restaurants_name".title(): names, "url": child_urls})
    
    df.to_csv('yelp_dataset.csv', mode = 'a', index = False, header = False)

# # Order: "reservations" ,"delivery", "burgers", "chinese", "japanese", "mexican", "italian",  "thai", "japanese"
# # Problem: "japanese", "mexican", "italian",

businesses = [
             "mexican", "italian","thai",
             'autorepair', 'car_dealers', 'auto_detailing', 
             'dryclean', 'hair', 'mobilephonerepair', 'gyms', 'bars', 'massage', 'nightlife', 'shopping'
            ]

for business in businesses:   
    get_urls(business)