# hejas Wojtku

zestaw1 = [1, 2, 5, 3, 2, 3]
zestaw2 = ['Mania', 'Wojti', 'Boguś', 'Włodziu', 'Kevin']

nowa_rzecz = []

for dana in zestaw2:
# ten kod wykona się tyle razy, ile jest elemenow w 'zestaw1':
# przy kazdym wykonaniu pod zmienna dana podstawiony zostanie element z zestawu1
    nowa_rzecz.append(dana + str(1))
# aż dotąd
print(nowa_rzecz)

nowa_rzecz = []
for (indx, dana) in enumerate(zestaw2):
    nowa_rzecz.append(dana + str(indx))
print(nowa_rzecz)

################### dotąd było ok

#### tera jest fuckup. :)
for a in range(len(zestaw2)):
    zestaw2[a] = "NoCOjeSt!?"
print(zestaw2)

#### co to kufa jest ten range()?

nasz_range = range(5)
print( nasz_range )
# [0, 1, 2, 3, 4] <- możesz myśleć że to 'tak jakby' jest nasz range
lista_z_range = list(nasz_range)
print(lista_z_range)

print(list(range(2, 114, 3)))
print(list(range(3, 114, 3)))

# range(start=0, stop, step=1)
# range(start, stop, step)

for a in range(7):
# for a in range(0, 7, 1):
# for a in [0,1,2,3,4,5,6]
    print(a)
######## I TERA JAAASNE :P
print()

zestaw2 = ['Mania', 'Wojti', 'Boguś', 'Włodziu', 'Kevin']
dl_zestaw_danych2 = len(zestaw2)
print('dlugosc zestawu 2:', dl_zestaw_danych2)

for i in range(dl_zestaw_danych2):
# for i in range(0, dl_zestaw_danych2, 1):
# for i in [0, 1, 2, 3, 4]:
    print(zestaw2[i])

print('\n')
print("Co drugie imie:")
# range(start=0, stop, step=1)
for i in range(0, len(zestaw2), 2):    
    print(zestaw2[i])