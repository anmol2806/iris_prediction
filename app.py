#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib


app = Flask(__name__)
# load the saved model file and use for prediction
logit_model = joblib.load('logit_model.pkl')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
       
    prediction=logit_model.predict(final_features)
    
   
    return render_template('preview.html', pred= prediction)

if __name__ == '__main__':
    app.run(debug=False)
