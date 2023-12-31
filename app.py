import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__, static_folder='static')
## Load the model
regmodel=pickle.load(open('xgboostmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']  # Access the 'data' key in JSON directly
        print("Received data:", data)

        # No need to convert the values again as they are already of type float
        input_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = regmodel.predict(input_data)
        print("Prediction:", output[0])

        return jsonify(float(output[0]))  # Convert prediction to float for JSON serialization

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     