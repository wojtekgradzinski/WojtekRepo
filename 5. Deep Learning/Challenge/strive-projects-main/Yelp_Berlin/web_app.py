import streamlit as st
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange
from time import sleep

st.title('Target Businesses: Report')

df_progress = pd.read_csv('yelp_dataset.csv')
scraping_left = sum([True for idx,row in df_progress.iterrows() if any(row.isnull())])


while scraping_left > 0:

    df_progress = pd.read_csv('yelp_dataset.csv')
    scraping_left = sum([True for idx,row in df_progress[['address','review_count']].iterrows() if any(row.isnull())])

    st.write("Scraping progress: ", scraping_left, "left off", df_progress.shape[0], "businesses")

    refresh = 30
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(refresh):
# Update the progress bar with each iteration.
        latest_iteration.text(f'Refreshing in {refresh - i + 1} seconds...')
        bar.progress(i + 1)
        sleep(1)
    









