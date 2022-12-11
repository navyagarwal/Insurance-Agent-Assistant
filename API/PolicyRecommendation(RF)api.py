import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

transactions = pd.read_csv('Transaction_Data.csv')
policies = pd.read_csv('Policy_Info.csv')


label_encoder = preprocessing.LabelEncoder()
transactions['Gender']= label_encoder.fit_transform(transactions['Gender'])
transactions['Residence']= label_encoder.fit_transform(transactions['Residence'])

transactions['PolicyName']= transactions['PolicyName'].str.replace("Policy_", "").astype("int")

features = []
for i in range(1, len(transactions.columns) - 1):
    features.append(transactions.columns[i])

X = transactions.loc[:, features]
y = transactions.loc[:, ["PolicyName"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, train_size = .75)

rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train.values.ravel())

from flask_ngrok import run_with_ngrok
from flask import Flask

app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when app is run

@app.route('/<int:age>/<int:gender>/<int:income>/<int:residence>/<int:diabetes>/<int:heart>/<int:bp>/<int:surg>/<int:covid>')
def predictPolicy(age, gender, income, residence, diabetes, heart, bp, surg, covid):
  return str("Policy ", rf_clf.predict([[age, gender, income, residence, diabetes, heart, bp, surg, covid]]))

app.run()
