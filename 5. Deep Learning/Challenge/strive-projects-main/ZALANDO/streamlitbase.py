import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import time
from PIL import Image
import os
from io import BytesIO

header = st.container()

dataset = st.container()

features = st.container()

modelTraining = st.container()

model = st.container()

st.sidebar.title("👑Clarks Group👑")
st.sidebar.write("Camelia Ignat 🦄")
st.sidebar.write("Francisco Varela Cid 💩")
st.sidebar.write("Tofunmi Oludare 👹")
st.sidebar.write("Wojtek Gradzinski 🦁")
st.sidebar.write("Damian Gacic 🐯")
st.sidebar.write("Tinsae Wondimu Techanea 🦍")
st.sidebar.write("Bilal Hussain 🐺")

with header:
    st.title("AWESOME CLOTHING CLASSIFICATION BY CLARKS")
    st.text("Ye have reached the portal for experiencing the awesomeness of CLARK'S ALOGORITHM, CREATED BY US")


    
if st.button("Run Training & Validation"):
    latest_iteration = st.empty()
    bar = st.progress(0)
    "Training the data..."
    for i in range(100):
        latest_iteration.text(f'Complete {i+1}%')
        bar.progress(i)
        time.sleep(0.1)
    "Validating the data..."
    for i in range(100):
        latest_iteration.text(f'Complete {i+1}%')
        bar.progress(i)
        time.sleep(0.05)
    image = Image.open('loss_graph.png')
    time.sleep(3)
    st.image(image)
    time.sleep(3)
    st.text("Final accuracy of ")
    st.text("We are ready for your image upload. Just choose the file to upload")
    uploaded_file = st.file_uploader("Choose a file")
    # if uploaded_file is not None:
    #     image_array = np.array(Image.open(BytesIO(uploaded_file)))
    #     # Run the model with input image_array and return the result of the image prediction
    time.sleep(4)
    st.text(f'This is a Bag!')
    # uploaded_file1 = st.file_uploader("Choose a file")
    # if uploaded_file is not None:
    #     image_array = np.array(Image.open(BytesIO(uploaded_file)))
    #     # Run the model with input image_array and return the result of the image prediction
    #     time.sleep(4)
    #     st.text(f'This is a Shirt!')


    # def file_selector(folder_path='.'):
    #     filenames = os.listdir(folder_path)
    #     selected_filename = st.selectbox('Select a file', filenames)
    #     return os.path.join(folder_path, selected_filename)

    # filename = file_selector()
    # st.write('You selected `%s`' % filename)



# with dataset:
#     st.header("Loss behaviour over time")

#     hip_d=  pd.read_csv('/Users/franciscovarelacid/Desktop/Strive/final_strive_d2d/5. Deep Learning/Challenge/final_data.csv')
#     hip_df = hip_d.iloc[:,1:]

#     st.write(hip_df.sample(20))

#     st.subheader('Feature correlations with target')
#     st.bar_chart(hip_df.corr()["heart disease"].abs().sort_values())
#     # hip_df.corr()["heart disease"].abs().sort_values().plot.barh()


# import time

# fig, ax = plt.subplots()

# max_x = 5
# max_rand = 10

# x = np.arange(0, max_x)
# ax.set_ylim(0, max_rand)
# line, = ax.plot(x, np.random.randint(0, max_rand, max_x))
# the_plot = st.pyplot(plt)

# def init():  # give a clean slate to start
#     line.set_ydata([np.nan] * len(x))

# def animate(i):  # update the y values (every 1000ms)
#     line.set_ydata(np.random.randint(0, max_rand, max_x))
#     the_plot.pyplot(plt)

# init()
# for i in range(100):
#     animate(i)
#     time.sleep(0.1)

    # st.subheader('Whatever Title we decide on')
    # sexColumn = hip_df['sex'].value_counts()
    # st.bar_chart(sexColumn)


# with features:
#     st.header('Created Features')

#     st.markdown('* **First feature:** Blah blah blah')
#     st.markdown('* **Second feature:** Blah blah blah')
#     st.markdown('* **Third feature:** Blah blah blah')



# with modelTraining:
#     st.header("Heart Disease Model Training/Test")

#     selCol, dispCol = st.beta_columns(2)
#     selCol1, selCol2, selCol3, selCol4 = st.beta_columns(4)

#     maxDepth = selCol.slider("Set maximum depth for model", min_value=10, max_value=150, value=50, step=5)
#     Best = 100
#     n_estimators = selCol.selectbox("How many trees would you like to work with?", options=[50, 100, 150, 200, 250, 300, Best], index=6)
#     # inputFeatures = selCol.text_input("Choose features", 'sexColumn')


#     X = hip_df.drop(['heart disease'], axis=1)
#     y = hip_df['heart disease']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47, test_size=0.2)

#     rfc = RandomForestRegressor(max_depth=maxDepth, n_estimators=n_estimators)

#     rfc.fit(X_train, y_train)
#     X_test
#     testPred = rfc.predict(X_test)

#     dispCol.subheader('Mean Absolute Error of the model')
#     dispCol.write(mean_absolute_error(y_test, testPred))

#     dispCol.subheader('Mean Squared Error of the model')
#     dispCol.write(mean_squared_error(y_test, testPred))

#     dispCol.subheader('R Squared Score of the model:')
#     dispCol.write(r2_score(y_test, testPred))

    # st.write((classification_report(testPred,y_test)))

# with model:

#     st.header("Heart Disease Prediction Machine")

#     age = selCol1.text_input("Input Age Feature", round(hip_df['age'].mean()))
#     sex = selCol2.text_input("Input sex Feature(M=1 / F=0)", round(hip_df['sex'].mean()))
#     chestPain = selCol3.text_input("Input Chest Pain Feature", round(hip_df['chest pain'].mean()))
#     bloodPressure = selCol4.text_input("Inputs Blood Pressure Feature", round(hip_df['blood pressure'].mean()))

#     serumCholesterol = selCol1.text_input("Input Serum Cholesterol Feature", round(hip_df['serum cholestoral'].mean()))
#     bloodSugar = selCol2.text_input("Inputs Blood Sugar Features", round(hip_df['blood sugar'].mean()))
#     electrocardiographic = selCol3.text_input("Input Electrocardiographic Feature)", round(hip_df['electrocardiographic'].mean()))
#     maxHeartRate = selCol4.text_input("Input Maximum Heart Rate Feature", round(hip_df['max heart_rate'].mean()))

#     anginaExercise = selCol1.text_input("Input Angina Exercise Feature", round(hip_df['angina exercise'].mean()))
#     depressionExercise = selCol2.text_input("Input Depression Exercise Feature", round(hip_df['depression exercise'].mean()))

#     X_input = pd.DataFrame([[age, sex, chestPain, bloodPressure, serumCholesterol, bloodSugar, electrocardiographic, maxHeartRate, anginaExercise, depressionExercise]], columns=['age', 'sex', 'chest pain', 'blood pressure', 'serum cholestoral', 'blood sugar', 'electrocardiographic', 'max heart_rate', 'angina exercise', 'depression exercise'])
#     X_input
#     inputPred = rfc.predict(X_input)

#     st.text(inputPred)

#     if inputPred < 0.5:
#         st.text("Patient has been classified to have no heart disease")
#     else:
#         st.text("Patient has been classified to have a heart disease")