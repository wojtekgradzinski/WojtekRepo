import requests
from bs4 import BeautifulSoup
import pandas as pd
page = requests.get("https://weather.com/weather/tenday/l/San+Francisco+CA?canonicalCityId=dfdaba8cbe3a4d12a8796e1f7b1ccc7174b4b0a2d5ddb1c8566ae9f154fa638c")

soup = BeautifulSoup(page.content, "html.parser")

'''
temperatures = soup.find_all('span', class_= "DailyContent--temp--_8DL5")

temperatures_text = []
for t in temperatures:
    temperatures_text.append(t.text)


names = soup.find_all('h2', class_= "DetailsSummary--daypartName--1Mebr")

names_text = []
for n in names:
    names_text.append(n.text)
'''

day_template = "detailIndex"

df = pd.DataFrame({},columns = ['day','Temp_u','temp_l'])

for i in range(15):
     
    day = soup.find('div', id=day_template+str(i) )

    name = day.find('h2', class_= "DetailsSummary--daypartName--1Mebr").text

    temp_u = day.find('span', class_ = "DetailsSummary--highTempValue--3x6cL").text[:2]
    temp_l = temp = day.find('span', class_ = "DetailsSummary--lowTempValue--1DlJK").text[:2]
    print("Day:", name, " The temperatures will be:", temp_u ,"/",temp_l )