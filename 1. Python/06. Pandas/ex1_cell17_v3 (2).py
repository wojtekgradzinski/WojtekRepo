# I decided to beat the exercise from the workbook!

import requests
import pandas as pd
from bs4 import BeautifulSoup

page = requests.get("https://forecast.weather.gov/MapClick.php?lat=37.777120000000025&lon=-122.41963999999996#.X9DVpBakolQ")
soup = BeautifulSoup(page.content, 'html.parser')

# first filtering we do over <div class="tombstone-container"> because it suits our needs
all_data = soup.findAll('div', class_="tombstone-container")

# lists for storage
period_names = []
short_descriptions = []
temperatures = []

# fill them with data :)
for data in all_data:
    period_names.append(data.find('p', class_='period-name').get_text(separator=' '))
    short_descriptions.append(data.find('p', class_='short-desc').get_text(separator=' '))    
    temperatures.append(data.find('p', class_='temp').get_text())

# let's zip them into touples!
data_touples = zip(period_names, short_descriptions, temperatures)

# and create a nice data frame :)
weather_data = pd.DataFrame(data_touples, columns=['period-name', 'short-desc', 'temp'])


#################################
# and with dedication for Wojtek:
def to_celsius(fahrenheit):
    i_fahr = int(fahrenheit)
    d_cels = round((i_fahr - 32) * 5/9, 1)
    return str(d_cels)


temperatures_c = []
for t in temperatures:
    t_splitted = t.split()
    t_splitted[1] = to_celsius(t_splitted[1])
    t_splitted[2] = t_splitted[2].replace('F', 'C')
    t_joined = " ".join(t_splitted)
    temperatures_c.append(t_joined)

data_touples = zip(period_names, short_descriptions, temperatures_c)
weather_data_c = pd.DataFrame(data_touples, columns=['period-name', 'short-desc', 'temp'])


print(weather_data)
print()
print(weather_data_c)