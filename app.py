from flask import Flask, render_template
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    df = pd.read_csv('health_data.csv')
    model = joblib.load('anomaly_model.pkl')
    df['anomaly'] = model.predict(df[['heart_rate', 'blood_oxygen']])
    df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    data = df.tail(1).to_dict('records')[0]
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
