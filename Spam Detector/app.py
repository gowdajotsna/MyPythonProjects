#Import main library
import numpy as np
import flask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 
#Import Flask modules
from flask import Flask, request, render_template

#Import pickle to save our regression model
import pickle 

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'template')

#Open our model 
model = pickle.load(open('svmspammodel.pkl','rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
#create our "home" route using the "index.html" page
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('SpamUI.html')

#Set a post method to yield predictions on page
@app.route('/predict', methods = ['POST'])
def predict():

    #obtain all form values and place them in an array, convert into integers
    int_features = [request.form['EmailInput']]
    # print("heloooooooooooooooooooooo",str(int_features))
    #Combine them all into a final numpy array
    # final_features = [np.array(int_features)]
    print(int_features)
    inputdatafeaturez = cv.transform(int_features)
    #predict the price given the values inputted by user
    prediction = model.predict(inputdatafeaturez)
    print(str(prediction))
    # #Round the output to 2 decimal places
    # output = round(prediction[0], 2)
    
    # #If the output is negative, the values entered are unreasonable to the context of the application
    # #If the output is greater than 0, return prediction
    # if output < 0:
    return render_template('SpamUI.html', prediction_text = prediction[0])
    # elif output >= 0:
    #     return render_template('index.html', prediction_text = 'Predicted Price of the house is: ${}'.format(output))   

#Run app
if __name__ == "__main__":
    app.run(debug=True)