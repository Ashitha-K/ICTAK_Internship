from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('lr.pkl', 'rb') as f:
    model= pickle.load(f)
with open('encoder.pkl','rb') as encoder:
          encoder=pickle.load(encoder)
with open ('scaler.pkl','rb') as scaler:
    scaler=pickle.load(scaler)
#with open ('imputer.pkl','rb') as imputer:
     #imputer=pickle.load(imputer)
#with open ('scaler.pkl','rb') as scaler_1:
    #scaler_1=pickle.load(scaler_1)


@app.route('/')
def home():
    return render_template('first1.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]

    final_features = np.array(features).reshape(1, -1)
    data=scaler.transform(final_features)
    #print(f"Features: {features}")
    #print(f"Final Features: {final_features}")

    prediction = model.predict(data)
    #return render_template('first.html', prediction_text='Predicted  score is: ${:.2f}'.format(prediction[0]))
    return f"<h1>Satisfaction score is: {prediction}</h1>"

if __name__ == '__main__':
    app.run(debug=True)