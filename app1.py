# -*- coding: utf-8 -*-

 #-*- coding: utf-8 -*-

#from flask import Flask, render_template

#app = Flask(__name__)

#@app.route('/')
#def home():
   #return render_template('index.html')  # Flask looks for index.html in the 'templates' folder

#if __name__ == '__main__':
   #app.run(debug=True)


from flask import Flask, request, render_template
import joblib

# Load the trained Logistic Regression model and vectorizer
model = joblib.load('logistic_model2.pkl')
vectorizer = joblib.load('vectorizer2.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the form
    user_input = [request.form['tweet']]  # Expecting a text field with the name 'tweet'

    # Vectorize the input text
    vectorized_input = vectorizer.transform(user_input)

    # Make prediction
    prediction = model.predict(vectorized_input)[0]

    # Map the prediction to the corresponding label
    labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    result = labels[prediction]

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)