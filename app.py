#importing the libraries
import pickle
import flask
from flask import Flask,render_template,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os 

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
load_tfidf = pickle.load(open('tfidfvect2.pkl', 'rb'))

def fake_news_det(news_headline):
	vectorized_input_data = load_tfidf.transform([news_headline]).toarray()
	prediction = model.predict(vectorized_input_data)
	if prediction == 0:
		return "This news is unverified, beware of what action you will take on it"
	else:
		return "This is a verified news, follow page for more verified news"


	return prediction

@app.route("/")

def home():
	return render_template('index.html')

@app.route("/predict", methods = ["POST", 'GET'])
def predict():
	if request.method == "POST":
		predict = request.form["news_headline"]
		pred = fake_news_det(predict)

	
	return render_template("index.html", prediction = pred)



if __name__ == "__main__":
	app.run(debug=True)
