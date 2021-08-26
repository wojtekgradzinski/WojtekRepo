import requests
from bs4 import BeautifulSoup
import pandas as pd


page = requests.get("https://www.imdb.com/list/ls091520106/?st_dt=&mode=detail&page=1&sort=user_rating,desc&ref_=ttls_ref_gnr&genres=Adventure")

soup = BeautifulSoup(page.content, 'html.parser')

all_data = soup.find_all('div', class_ = "lister-item-content")


# Lists for data Storage
movie_name = []
desc = []
release_date = []
director = []
rating = []
duration = []
genre = []
actors = []
filming_dates = []
gross = []


######## Fill lists with extracted data ########


for data in all_data:
    movie_name.append(data.find('a').get_text(separator=' '))
    desc.append(data.find('p', class_= "").get_text(separator=' ')[5:])
    release_date.append(int(data.find('span', class_="lister-item-year text-muted unbold").get_text(separator=' ').strip('()I ')))
    rating.append(data.find('span', class_ = 'ipl-rating-star__rating').get_text(separator=' '))
    duration.append(int(data.find('span', class_ = 'runtime').get_text(separator=' ')[:-4]))
    genre.append(data.find('span', class_ = 'genre').get_text(separator=' ')[1:-12].split(', '))



people = soup.find_all('p', class_ ='text-muted text-small')

for person in people[1::3]:
    person1= person.get_text(separator=' ')
    splittedPerson1 = person1.split('|')
    splittedPerson1a = splittedPerson1[0][16:]
    splittedPerson1b = splittedPerson1[1][15:]
    final_actors = splittedPerson1b.replace(' \n', '')
    final_director = splittedPerson1a.replace(' \n', '')
    director.append(final_director[:-1].split(' , '))
    actors.append(final_actors.split(' , '))
    


values = soup.find_all('p', class_ = 'text-muted text-small')

for value in values[2::3]:
    raw_text = value.get_text(separator=' ')
    values_splitted = raw_text.split('|')
    gross_v = values_splitted[1].strip('Gross: ')
    gross_f = gross_v.replace('$', '').replace('M', '')
    gross.append(float(gross_f.replace('\n ', '').replace(' \n', '')))


######## Create Pandas DataFrame ########

dataList = zip(movie_name, genre, desc, release_date, director, actors, rating, duration, gross)
movieData = pd.DataFrame(dataList, columns = ['Title', 'Genre', 'Summary', 'Release Date', 'Directors', 'Actors', 'Ratings', 'Duration', 'Gross Income'])


print(movieData)
