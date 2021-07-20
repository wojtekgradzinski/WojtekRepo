import streamlit as st
from PIL import Image # Required to show images
import math
# import matplotlib.pyplot as plt
# import seaborn as sns

import numpy    as np
import pandas   as pd
from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config
from sklearn.model_selection import train_test_split


from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier

df = pd.read_csv('./data/data.csv')

df = df.drop(['slope','ca','thal'],axis=1)
df = df.replace('?', np.nan)

cat_vars  = ['sex', 'cp','fbs','restecg', 'exang']         
num_vars  = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


df = df.apply(pd.to_numeric)
X = df.drop('num       ',axis =1)
y = df['num       ']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

num_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='mean')),
])

cat_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value= -1)),
    ('ordinal', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)) #  ONLY IN VERSION 0.24
])

tree_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_4_treeModels, num_vars),
    ('cat', cat_4_treeModels, cat_vars),
], remainder='drop')

tree_classifiers = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":   ExtraTreesClassifier(n_estimators=100),
  "Random Forest": RandomForestClassifier(n_estimators=100),
  "AdaBoost":      AdaBoostClassifier(n_estimators=100),
  "Skl GBM":       GradientBoostingClassifier(n_estimators=100),
  "Skl HistGBM":   HistGradientBoostingClassifier(max_iter=100),
  "XGBoost":       XGBClassifier(n_estimators=100),

}
tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}
results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': []})
for model_name, model in tree_classifiers.items():
    
    
    model.fit(X_train, y_train)
    #start_time = time.time()
    pred = model.predict(X_test)

    #total_time = time.time() - start_time
            
    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100},ignore_index=True)

results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 

results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['Accuracy', 'Bal Acc.'], vmin=0, vmax=100, color='#5fba7d')

best_model = tree_classifiers[results_ord.iloc[0].Model]
best_model.fit(X_train,y_train)




header_container = st.beta_container()
result = st.beta_container()

with header_container:
    st.image('data/data/header.jpg')
    st.title("We make sure your heart doesn't hurt!")

logo = Image.open("data/data/logo.jpg")
st.sidebar.image(logo, width=200)

rad = st.sidebar.radio('Navigation',['About us','Feature Engineering', 'prediction pipeline', 'User input'])

if rad == 'About us':
    st.header("About:")
    st.markdown("""*A team named after Aelius Galen, A Greek physician surgeon and philosopher from the roman empire.*""")
    st.markdown("***Task:***")
    st.markdown("Predicting a heart attack given a few data samples and with a constraint of avoiding overdoes")
    st.markdown("***Team Members:***")
    st.markdown('* *Gozal*')	
    st.markdown('* *Rajat *')	
    st.markdown('* *Tinsae*')	

elif rad == 'Feature Engineering':
    st.markdown("## Feature Engineering")

    st.markdown("Corelation of features")
    st.image('data/99.PNG')

    st.markdown("Age vs Cholesterol: Higher Age is more likely to have Heart Attack, although no correlation to cholesterol level")
    st.image('data/Agevs.PNG')
    
    st.markdown("Maximum Heart Rate vs Cholesterol. Surprisingly no correlation, as the spread of result is in all direction")
    st.image('data/2.PNG')

    st.markdown(" Resting Blood Pressure vs Maximum Heart Rate. Higher Heart Rate is more likely to have Heart Attack, but not really related to Resting Blood Pressure")
    st.image('data/3.PNG')

    st.markdown(" Resting Blood Pressure vs Maximum Heart Rate. Higher Heart Rate is more likely to have Heart Attack, but not really related to Resting Blood Pressure")
    st.image('data/4.PNG')

    # c1, c2, c3= st.beta_columns(3)
    # c1.image('data/11.PNG')
    # c2.image('data/12.PNG')
    # c3.image('data/13.PNG')


elif rad == 'prediction pipeline':
    st.markdown("## Prediction pipeline")
    

elif rad == 'User input':
    st.markdown("## User input")
    col1, col2 = st.beta_columns(2)

    gender_in = col1.selectbox('Select your gender', options=['Female','Male']) 
    if gender_in == 'Female':
        gender = 0
    else:
        gender =1
    
    age_in = col1.slider('Age', min_value = 1, max_value= 110, value=20, step = 10)
    age = int(age_in)

    chest_pain_in = col1.selectbox('Select chest pain type', options=['Typical angina ','Atypical angina', 'Non-anginal pain','Asymptomatic']) 
    if chest_pain_in == 'Typical angina ':
        cp = 1
        
    elif chest_pain_in == 'Atypical angina':
        cp = 2
    elif chest_pain_in == 'Non-anginal pain':
        cp =3
    elif chest_pain_in == 'Asymptomatic':
        cp = 4

    trestbps_in = col1.slider('Resting blood pressure', min_value = 100, max_value= 200, value=110, step = 10)
    trestbps = int(trestbps_in)

    chol_in = col1.text_input('Input serum cholestoral leven in mg/dl', 200)
    chol = int(chol_in)

    fbs_in = col2.selectbox('fasting blood sugar > 120 mg/d', options=['Yes','No'])
    if fbs_in == 'Yes':
        fbs = 1
    else:
        fbs = 0
    

    restecg_in	 = col2.selectbox('Resting electrocardiographic results', options=['Normal ','having ST-T wave abnormality', 'left ventricular hypertrophy']) 
    if restecg_in == 'Normal ':
        restecg = 0
    if restecg_in == 'having ST-T wave abnormality':
        restecg = 1
    if restecg_in =='left ventricular hypertrophy':
        restecg =1

    thalach_in = col2.slider('maximum heart rate achieved ', min_value = 100, max_value= 200, value=160, step = 10)
    thalach = int(thalach_in)
    exang_in = col2.selectbox('exercise induced angina', options=['Yes','No'])
    if exang_in =='Yes':
        exang = 1
    else:
        exang = 0

    oldpeak_in = col2.text_input('ST depression induced by exercise relative to rest ', 3.0)
    oldpeak = float(oldpeak_in)

x = pd.DataFrame([{'age':age, 'sex':gender , 'cp':cp, 'trestbps':trestbps,	'chol' :chol,	'fbs':fbs,	'restecg':restecg,	'thalach':restecg,	'exang':restecg, 'oldpeak':oldpeak}])
d = best_model.predict(x)
if d[0] == 1:
    s = 'You might have heart disease'
    st.error(s)
else:
    s = 'You do not have heart disease'
    st.error(s)
    st.balloons()

# st.write(s)
# st.error("Do you really, really, wanna do this?")
# if st.button("Yes I'm ready to rumble"):
#     d[0]==0