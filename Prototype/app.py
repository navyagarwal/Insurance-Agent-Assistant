import pandas as pd
import numpy as np
from math import *
import tensorflow as tf
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

transactions = pd.read_csv('Transaction_Data2.csv')
policies = pd.read_csv('Policy_Info.csv')


label_encoder = preprocessing.LabelEncoder()
transactions['Gender']= label_encoder.fit_transform(transactions['Gender'])
# transactions['Residence']= label_encoder.fit_transform(transactions['Residence'])

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
app.config['SECRET_KEY'] = 'xyzflaskapp'


@app.route('/')
def hello():
  # return str("Welcome Visiter")
  return render_template('expr.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
  form = profileForm()
  if form.is_submitted():
    age = form.age.data
    gender = form.gender.data
    income = form.income.data
    # residence = form.residence.data
    # diabetes = form.diabetes.data
    # heart = form.heart.data
    # bp = form.bp.data
    # other = form.other.data
    # surg = form.surg.data
    # covid = form.covid.data
    # pred = rf_clf.predict([[age, gender, income, residence, diabetes, heart, bp, other, surg, covid]])[0]
    pred = rf_clf.predict([[age, gender, income]])[0]
    pol1 = 'Policy_' + str(pred)
    topTwo = nextTwo(pol1)
    pol2 = topTwo[0]
    pol3 = topTwo[1]

    # with open ("Policy_Info.csv", "r") as source:
    #   reader = csv.reader(source)

    # policies = pd.read_csv('Policy_Info.csv')

    # row1 = policies.loc[pol1]
    # row2 = policies.loc[pol2]
    # row3 = policies.loc[pol3]

    # pName1=row1['Name'] 
    # pIns1=row1['Insurer']
    # pCov1=row1['Cover(lac)']
    # pPrem1=row1['Premium(annual)']
    # pWait1=row1['Pre-Existing Waiting Period']
    # pClaim1=row1['ClaimSettlementRatio']
    # pMat1=row1['Maternity']
    # pOPD1=row1['OPD Benefits']

    # pName2=row2['Name'] 
    # pIns2=row2['Insurer']
    # pCov2=row2['Cover(lac)']
    # pPrem2=row2['Premium(annual)']
    # pWait2=row2['Pre-Existing Waiting Period']
    # pClaim2=row2['ClaimSettlementRatio']
    # pMat2=row2['Maternity']
    # pOPD2=row2['OPD Benefits']

    # pName3=row3['Name'] 
    # pIns3=row3['Insurer']
    # pCov3=row3['Cover(lac)']
    # pPrem3=row3['Premium(annual)']
    # pWait3=row3['Pre-Existing Waiting Period']
    # pClaim3=row3['ClaimSettlementRatio']
    # pMat3=row3['Maternity']
    # pOPD3=row3['OPD Benefits']


    # return render_template('page2.html', pred1 = pol1, pred2 = pol2, pred3 = pol3, pName1=pName1, pIns1=pIns1, pCov1=pCov1, pPrem1=pPrem1, pWait1=pWait1, pClaim1=pClaim1, pMat1=pMat1, pOPD1=pOPD1,
    # pName2=pName2, pIns2=pIns2, pCov2=pCov2, pPrem2=pPrem2, pWait2=pWait2, pClaim2=pClaim2, pMat2=pMat2, pOPD2=pOPD2,
    # pName3=pName3, pIns3=pIns3, pCov3=pCov3, pPrem3=pPrem3, pWait3=pWait3, pClaim3=pClaim3, pMat3=pMat3, pOPD3=pOPD3)

    return render_template('page2.html', pred1 = pol1, pred2 = pol2, pred3 = pol3)

  # return str("hi")
  return render_template('Form1.html', form=form)


if __name__ == '__main__':
    app.run()
