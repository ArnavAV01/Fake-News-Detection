from flask import Flask, render_template, request
from joblib import load
from preprocess import preprocess_and_vectorize, predict_news_category

app = Flask(__name__)

model = load("news_classification.joblib")

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get the input news article from the form
        news_article = request.form['news_article']
        
        # Perform prediction using the loaded model
        prediction = predict_news_category(news_article, preprocess_and_vectorize, model)

        # Render the form to input the news article along with the prediction result
        return render_template('index.html', prediction=prediction)
    
        # Render the form to input the news article
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
   app.run(debug=True)
