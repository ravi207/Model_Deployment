from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from sklearn import linear_model
#from sklearn.externals import joblib
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
  

import flask
app = Flask(__name__)
clf = joblib.load('quora_model.pkl')
count_vect = joblib.load('quora_vectorizer.pkl')
    
###################################################
def pre_processing(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub('[0-9]+','num',text)
    word_list = nltk.word_tokenize(text)
    word_list =  [lemmatizer.lemmatize(item) for item in word_list]
    return ' '.join(word_list)
###################################################


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    review_text = pre_processing(to_predict_list['review_text'])
    
    pred = clf.predict(count_vect.transform([review_text]))
    prob = clf.predict_proba(count_vect.transform([review_text]))
    #pr =  1
    if prob[0][0]>=0.5:
        prediction = "Positive"
        #pr = prob[0][0]
    else:
        prediction = "Negative"
        #pr = prob[0][0]

    # sanity check to filter out non questions. 
    if not re.search("(?i)(what|which|who|where|why|when|how|whose|\?)",to_predict_list['review_text']):
        prediction = "Negative"
        #prob = prob*0
        
   
    
    return flask.render_template('predict.html', prediction = prediction, prob =np.round(prob[0][0],3)*100)


if __name__ == '__main__':
    #clf = joblib.load('quora_model.pkl')
    #count_vect = joblib.load('quora_vectorizer.pkl')
    app.run(debug=True)
    #app.run(host='localhost', port=8081)
