import streamlit as st
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange
from time import sleep

# st.title('Target Businesses: Report')

# df_progress = pd.read_csv('yelp_dataset.csv')
# scraping_left = sum([True for idx,row in df_progress.iterrows() if any(row.isnull())])


# while scraping_left > 0:

#     df_progress = pd.read_csv('yelp_dataset.csv')
#     scraping_left = sum([True for idx,row in df_progress[['address','review_count']].iterrows() if any(row.isnull())])

#     st.write("Scraping progress: ", scraping_left, "left off", df_progress.shape[0], "businesses")

#     refresh = 30
#     latest_iteration = st.empty()
#     bar = st.progress(0)

#     for i in range(refresh):
# # Update the progress bar with each iteration.
#         latest_iteration.text(f'Refreshing in {refresh - i + 1} seconds...')
#         bar.progress(i + 1)
#         sleep(1)


rest_win_subcategory = ['Beach Bars', 'Mexican', 'Bavarian', 'Cafeterias', 'Himalayan/Nepalese']
other_win_subcategory = 'Zoo'
count_businesses = 755


st.title('Best Your Momma VC Investment in Berlin by DataManiacs')

st.write("""

The Team

-Francisco

-Bartosz

-Rajat

-Mateo



""")
st.write('How many businesses did we take into consideration? *{count_businesses}*'.format(count_businesses=count_businesses))
st.subheader('We will present the best option for each main subcategory')

data_frame1 = pd.read_csv('data/shit.csv')
data_frame2 = pd.read_csv('data/other.csv')
data_frame_corr = pd.read_csv('data/insignificant_correlations.csv')

if st.button("For Restaurants"):
    st.subheader('Winning sub-category: ')
    st.write('{rest_win_subcategory}'.format(rest_win_subcategory=rest_win_subcategory))
    st.write(data_frame1)
    st.subheader('Key Factors for success: ')
    st.write('There was no correlation')
    st.write(data_frame_corr)
    # st.write('Days Open: {rest_days}'.format(rest_days=rest_days))
    # st.write('Days Open: {rest_att}'.format(rest_att=rest_att))


if st.button("For Other Business Types"):
    st.subheader('Winning sub-category: ')
    st.write('We can see that generally there are happy with the service but if we could choose one we would choose a Zoo')
    st.write('{other_win_subcategory}'.format(other_win_subcategory=other_win_subcategory))
    st.write(data_frame2)
    st.subheader('Key Factors for success: ')
    st.write('Not enough data to take significant conclusion')

    # st.write('Days Open: {other_days}'.format(other_days=other_days))
    # st.write('Days Open: {other_att}'.format(other_att=rest_att))

if st.button("Assumptions"):
    st.write("""Using the Pareto Law we looked for:
    
    + The worst rated categories - High Opportunity of Improvement (unhappy clients)

    + The categories with the most reviews - High Demand

    + The categories with the least businesses - Low Supply
    
    """)
