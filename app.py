import pickle5 as pickle
from flask import Flask, request, app, render_template
import numpy as np
import pandas as pd



#preprocessed_data_path = os.path.join("src","notebook","saved_picklefile","scale_numeric_var.pkl")
with open("scale_numeric_var.pkl","rb") as f:
    transform_data = pickle.load(f)

#model_path  = os.path.join("src","notebook","saved_picklefile","linear_model.pkl")
with open("linear_model.pkl",'rb') as f1:
    knnmodel= pickle.load(f1)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('index.html')
    else:
        data = request.form
        input_data_arr = np.array(list(data.values()))
        transfomred_input_data = transform_data.transform(input_data_arr.reshape(1,-1))

        print(data)
        print(input_data_arr)
        print(transfomred_input_data)
        
        results = knnmodel.predict(transfomred_input_data)
        print(results)
    return render_template('index.html',results = results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
