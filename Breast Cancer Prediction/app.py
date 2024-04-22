from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Collecting input data and storing it in a DataFrame
        input_data = pd.DataFrame({
            'compactness_mean': [float(request.form['compactness_mean'])],
            'concavity_mean': [float(request.form['concavity_mean'])],
            'concave points_mean': [float(request.form['concave points_mean'])],
            'symmetry_mean': [float(request.form['symmetry_mean'])],
            'fractal_dimension_mean': [float(request.form['fractal_dimension_mean'])],
            'texture_se': [float(request.form['texture_se'])],
            'symmetry_se': [float(request.form['symmetry_se'])],
            'texture_worst': [float(request.form['texture_worst'])],
            'area_worst': [float(request.form['area_worst'])],
            'compactness_worst': [float(request.form['compactness_worst'])],
            'concavity_worst': [float(request.form['concavity_worst'])],
        })

        # Scaling input data
        input_data_scaled = scaler.transform(input_data)
        new_input_data = pd.DataFrame(input_data_scaled,columns=input_data.columns)
        
        # Make prediction
        prediction = model.predict(new_input_data)
        
        if prediction == 1:
            prediction = 'Malignant'
        elif prediction == 0:
            prediction = 'Benign'
        
        return render_template('home.html', prediction=prediction)
    
    return render_template('home.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)


