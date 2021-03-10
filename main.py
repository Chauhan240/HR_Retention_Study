# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from flask import Flask, render_template, request,session, redirect, Markup
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from flask_mail import Mail
import json
import os
import math
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


with open('config.json','r') as c:
    params = json.load(c)["params"]

local_server=True
app=Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['UPLOAD_FOLDER'] = params['upload_location']
app.config.update(
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = '465',
    MAIL_USE_SSL = 'TRUE',
    MAIL_USERNAME = params['gmail-user'],
    MAIL_PASSWORD = params['gmail-password']
)
mail=Mail(app)

@app.route("/",methods= ['GET','POST']) #endpoint
def home():
    if (request.method == 'POST'):
        project=int(request.form.get('project'))
        hours=int(request.form.get('hours'))
        yr_spent=int(request.form.get('yr_spent'))
        n_accident=int(request.form.get('n_accident'))
        n_promotion = int(request.form.get('n_promotion'))
        evalu = float(request.form.get('evalu'))
        sat = float(request.form.get('sat'))
        dep = request.form.get('dep')
        salary = request.form.get('salary')
        input_user = np.array([[project,hours,yr_spent,n_accident,n_promotion,sat,evalu,dep,salary]])
        with open('impute.pkl', 'rb') as fp:
            imputation = pickle.load(fp)
        input_user[0,5:7] = imputation.transform(input_user[0,5:7].reshape(1,2))
        with open('OHE.pkl', 'rb') as fp:
            encode = pickle.load(fp)
        input_user=np.concatenate((input_user[:, 0:7], encode.transform(input_user[:,-2:]).toarray()), axis=1)
        with open('sc.pkl', 'rb') as fp:
            scaling = pickle.load(fp)
        input_user=scaling.transform(input_user)
        with open('model_lg.pkl', 'rb') as lg:
            model_lg = pickle.load(lg)
        with open('model_dt.pkl', 'rb') as dt:
            model_dt = pickle.load(dt)
        with open('model_dt_tune.pkl', 'rb') as dt_tune:
            model_dt_tune = pickle.load(dt_tune)
        with open('model_rf.pkl', 'rb') as rf:
            model_rf = pickle.load(rf)
        with open('model_bagc.pkl', 'rb') as bagc:
            model_bagc = pickle.load(bagc)
        with open('model_adb.pkl', 'rb') as adb:
            model_adb = pickle.load(adb)
        with open('model_gb.pkl', 'rb') as gb:
            model_gb = pickle.load(gb)
        with open('model_xgb.pkl', 'rb') as xgb:
            model_xgb = pickle.load(xgb)
        with open('model_knn.pkl', 'rb') as knn:
            model_knn = pickle.load(knn)
        v_prediction_text=''
        for model in [model_lg, model_dt, model_dt_tune, model_rf, model_bagc, model_adb, model_gb, model_xgb, model_knn]:
            pred = model.predict(input_user)
            if (int(pred[0]) == 1):
                v_prediction_text=v_prediction_text + ' <mark>' + str(type(model).__name__) + '</mark>  says : The employee will Leave the company <br> <br>'
            else:
                v_prediction_text = v_prediction_text  + ' <mark>' + str(type(model).__name__) + '</mark> says : The employee is happy to stay in the company <br> <br>'
        #pred = model_lg.predict(input_user)
        #if (int(pred[0]) == 1):
         #   v_prediction_text="The employee will Leave the company"
        #else:
         #   v_prediction_text = "The employee is happy to stay in the company: {} ".format(pred[0])
        return render_template('index.html', params=params, prediction_text= Markup(v_prediction_text))
    return render_template('index.html', params=params)

app.run(debug=True) # by adding debug = true the change wil be detected automatically and rerun by itself