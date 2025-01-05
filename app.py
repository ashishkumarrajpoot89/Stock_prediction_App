
from flask import Flask, render_template, request
from model import train_lstm_model
import plotly.graph_objs as go

app = Flask(__name__)

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']

        # Train the model and get predictions
        model, scaler, data = train_lstm_model(ticker)

        # Prepare data for prediction
        last_60_days = data['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Make prediction
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        # Plot the data using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price'))
        fig.add_trace(go.Scatter(x=[data.index[-1]], y=[predicted_price], mode='markers', name='Predicted Price'))
        graph_html = fig.to_html(full_html=False)

        return render_template('index.html', ticker=ticker, predicted_price=predicted_price, graph_html=graph_html)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
