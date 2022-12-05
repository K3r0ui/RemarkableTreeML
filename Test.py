import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import pymongo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OrdinalEncoder, OneHotEncoder
from sklearn import linear_model
st.write("Bonjour , on va tourner un algorithme pour prédire l'hauteur d'un arbre à partir du l'espece de l'arbre et l'arrondissement , sa stade de developpement et sa diametre :sunglasses:")

st.sidebar.header("INPUT")

def user_input():
    espece=st.sidebar.selectbox(
        "Selectionner une espece de l'arbre?",
        ('tulipifera', 'decurrens', 'simplex', 'cerris', 'grandiflora',
       'sylvatica', 'x hispanica', 'libani', 'orientalis', 'giganteum',
       'japonica', 'babylonica', 'hippocastanum', 'colurna', 'distichum',
       'nigra', 'bungeana', 'ilex', 'alba', 'baccata', 'biloba', 'robur',
       'fraxinifolia', 'paniculata', 'dioica', 'n. sp.', 'lotus',
       'ulmoides', 'virginiana', 'monspessulanum', 'glyptostroboides',
       'saccharinum', 'sempervirens', 'dulcis', 'bignonioides',
       'tomentosa', 'pseudoplatanus', 'glabra', 'koraiensis', 'carica',
       'stenoptera', 'cappadocicum', 'granatum', 'terebinthus',
       'azedarach', 'speciosa', 'pomifera', 'carpinifolia', 'frainetto',
       'suber subsp. Occidentalis', 'kaki', 'opalus', 'araucana',
       'coulteri', 'pseudoacacia', 'minor', 'sinensis',
       'nigra subsp. laricio', 'involucrata', 'pavia', 'australis'
    ))
    arrondissement=st.sidebar.selectbox(
        "Selectionner l'arrondissement de l'arbre?",
        ('BOIS DE VINCENNES', 'PARIS 8E ARRDT', 'PARIS 16E ARRDT',
       'PARIS 10E ARRDT', 'PARIS 7E ARRDT', 'PARIS 14E ARRDT',
       'PARIS 19E ARRDT', 'BOIS DE BOULOGNE', 'PARIS 20E ARRDT',
       'PARIS 9E ARRDT', 'PARIS 1ER ARRDT', 'PARIS 17E ARRDT',
       'PARIS 4E ARRDT', 'PARIS 13E ARRDT', 'PARIS 3E ARRDT',
       'PARIS 15E ARRDT', 'PARIS 5E ARRDT', 'PARIS 18E ARRDT',
       'PARIS 6E ARRDT', 'PARIS 12E ARRDT', 'PARIS 11E ARRDT')
    )
    stadedeveloppement=st.sidebar.selectbox(
        "Selectionner le stade de developpement?",
        ('Mature', 'Adulte', 'Jeune (arbre)')
    )
    circonferenceencm=st.sidebar.select_slider(
        "Selectionner le diametre de l'arbre?",
        options=[30.0,  53.0,  76.0,  85.0,  90.0, 105.0, 115.0, 118.0, 120.0, 130.0, 132.0,
       140.0, 145.0, 146.0, 150.0, 153.0, 155.0, 157.0, 160.0, 163.0, 164.0, 165.0,
       167.0, 169.0, 172.0, 173.0, 175.0, 180.0, 182.0, 186.0, 190.0, 195.0, 200.0,
       204.0, 205.0, 207.0, 209.0, 210.0, 211.0, 212.0, 215.0, 220.0, 221.0, 222.0,
       225.0, 230.0, 231.0, 232.0, 234.0, 237.0, 240.0, 241.0, 246.0, 250.0, 251.0,
       253.0, 255.0, 261.0, 265.0, 270.0, 285.0, 286.0, 289.0, 295.0, 299.0, 300.0,
       308.0, 310.0, 311.0, 315.0, 320.0, 321.0, 330.0, 335.0, 340.0, 347.0, 354.0,
       355.0, 358.0, 360.0, 365.0, 366.0, 368.0, 375.0, 380.0, 385.0, 390.0, 395.0,
       397.0, 400.0, 405.0, 407.0, 408.0, 410.0, 415.0, 420.0, 425.0, 426.0, 427.0,
       430.0, 431.0, 443.0, 445.0, 447.0, 450.0, 465.0, 468.0, 480.0, 490.0, 495.0,
       500.0, 502.0, 510.0, 511.0, 520.0, 534.0, 562.0, 583.0, 595.0, 597.0, 610.0,
       615.0, 634.0, 645.0, 695.0, 725.0]
    )
    data={
        'espece':espece,
        'arrondissement':arrondissement,
        'stadedeveloppement':stadedeveloppement,
        'circonferenceencm':circonferenceencm
    }
    input_params=pd.DataFrame(data,index=[0])
    return input_params
dfinput= user_input()
data={}
df = pd.DataFrame(data)
response_API = requests.get('https://opendata.paris.fr/api/records/1.0/search/?dataset=arbresremarquablesparis&q=&rows=175&facet=libellefrancais&facet=genre&facet=espece&facet=stadedeveloppement&facet=varieteoucultivar&facet=dateplantation')
#print(response_API.status_code)
r= response_API.json()
print (r['nhits'])
for i in range (r['nhits']):
  
  df = df.append(r['records'][i]['fields'],ignore_index=True)

st.subheader('Notre Dataset de base')
st.write(df)
st.subheader('On veut trouver lhauteur de cette arbre')
st.write(dfinput)
Dfmain= df[['espece','hauteurenm','arrondissement','stadedeveloppement','circonferenceencm']].sort_values(by = ['arrondissement'])
client = pymongo.MongoClient("mongodb+srv://krw:krw@cluster0.t2qz0z5.mongodb.net/?retryWrites=true&w=majority")
## POUR INSERER LA DATA BASE UNE AUTRE FOIS 
# data = Dfmain.to_dict(orient="records")
db = client["Machinelearning"]
# db.iris.insert_many(data)
val1 = Dfmain['espece'].value_counts()
val2 = Dfmain['arrondissement'].value_counts()
val3 = Dfmain['stadedeveloppement'].value_counts()
val4 = Dfmain['hauteurenm'].value_counts()
features_np = Dfmain[['espece','arrondissement','stadedeveloppement','circonferenceencm']].to_numpy()
encoder = OrdinalEncoder()
features_encoder=encoder.fit_transform(features_np)
x = pd.DataFrame(features_encoder)
y = Dfmain['hauteurenm']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)
regr = linear_model.LinearRegression()
regr.fit ( x_train , y_train )
array = dfinput.to_numpy().tolist()
arrayencoded = encoder.transform(array)
predictedCO2 = regr.predict(arrayencoded)
input = pd.DataFrame(dfinput)
# "hauteurenm"
pr = predictedCO2.tolist()
# predictedCO2 = pd.DataFrame(predictedCO2)
input['hauteurenm'] = pr
dataoutput= input.to_dict(orient="records")
db.iris.insert_many(dataoutput)
#print(predictedCO2)
# output = input.append(predictedCO2)

#print(type(dfinput.to_numpy().tolist()))
st.subheader('Le resultat de la prediction')
st.write('L HAUTEUR EST ',predictedCO2)

st.subheader('NOTRE Nouvelle arbre output est:')
st.write(input)
#==========================================
st.subheader('LISTES DES CHARTS')
st.write("L'hauteur en comparant par circonferance de l'arbre")
st.line_chart(Dfmain[['hauteurenm','circonferenceencm']])
st.write("nombre d'arbre par espece")
st.bar_chart(val1)
st.write("nombre d'arbre par arrondissement")
st.bar_chart(val2)
st.write("nombre d'arbre par status de developpement")
st.line_chart(val3)
st.write("nombre d'arbre par hauteur ")
st.area_chart(val4)
chart_data = pd.DataFrame(
    features_encoder,
    columns=['espece', 'arrondissement', 'stadedeveloppement','circonferenceencm'])

st.area_chart(chart_data)
