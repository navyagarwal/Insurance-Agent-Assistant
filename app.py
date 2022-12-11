import pandas as pd
import numpy as np
from math import *
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


label_encoder = preprocessing.LabelEncoder()
policies['Maternity']= label_encoder.fit_transform(policies['Maternity'])
policies['OPD Benefits']= label_encoder.fit_transform(policies['OPD Benefits'])

policies = policies.drop(['Name', 'Insurer'], axis=1)

for col in ['Cover(lac)', 'Premium(annual)', 'Pre-Existing Waiting Period', 'ClaimSettlementRatio']:
  policies[col] = (policies[col] - policies[col].min()) / (policies[col].max() - policies[col].min())

policies = policies.set_index('PolicyName')

def euclidean_dist(x, y):
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def similarityIn(polA, polB):
  lA = []
  lB = []
  for i in policies.loc[polA]:
    lA.append(i)
  for i in policies.loc[polB]:
    lB.append(i)
  return euclidean_dist(lA, lB)

def nextTwo(policy):
  dict = {}
  for pol in policies.index:
    dict[pol] = similarityIn(pol, policy)
  topOne = max(dict, key=dict.get)
  dict.pop(topOne)
  topTwo = max(dict, key=dict.get)
  dict.pop(topTwo)
  ans = [topOne, topTwo]
  return ans



#------------------------------------------------------


from flask import Flask, render_template, request, redirect, url_for
from form import profileForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'navyaflaskapp'


@app.route('/')
def hello():
  return str("Welcome Visiter")

@app.route('/predict/', methods = ['GET', 'POST'])
def predict():
  form = profileForm()
  if form.is_submitted():
    age = form.age.data
    gender = form.gender.data
    income = form.income.data
    residence = form.residence.data
    diabetes = form.diabetes.data
    heart = form.heart.data
    bp = form.bp.data
    other = form.other.data
    surg = form.surg.data
    covid = form.covid.data
    pred = rf_clf.predict([[age, gender, income, residence, diabetes, heart, bp, other, surg, covid]])[0]
    pol1 = 'Policy_' + str(pred)
    topTwo = nextTwo(pol1)
    pol2 = topTwo[0]
    pol3 = topTwo[1]
    return render_template('try3.html', pred1 = pol1, pred2 = pol2, pred3 = pol3)
  return render_template('try.html', form=form)


@app.route('/<int:age>/<int:gender>/<int:income>/<int:residence>/<int:diabetes>/<int:heart>/<int:bp>/<int:other>/<int:surg>/<int:covid>/')
def predictPolicy(age, gender, income, residence, diabetes, heart, bp, other, surg, covid):
  return render_template('try3.html', prediction = str(rf_clf.predict([[age, gender, income, residence, diabetes, heart, bp, other, surg, covid]])[0]))


@app.route('/name/<string:n>/')
def myName(n):
    return render_template('try2.html', name = n)

if __name__ == '__main__':
    app.run()