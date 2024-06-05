# import joblib
from joblib import load
import spacy
import gensim.downloader as api

nlp = spacy.load("en_core_web_sm")
# sample tweet text
# text = "Prime Minister Narendra Modi files nomination for Varanasi Lok Sabha seat with diverse support from astrology scholar Ganeshwar Shastri Dravid, Lalchand Kushwaha representing OBC segment, Sanjay Sonkar from Dalit community, and Baijnath Patel, reflecting broad social group backing."

# load the saved pipleine model
pipeline = load("news_classification.joblib")
wv = api.load("word2vec-google-news-300")

def preprocess_and_vectorize(text):
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return wv.get_mean_vector(filtered_tokens)

def predict_news_category(text, preprocess_and_vectorize, pipeline):
    # Preprocess and vectorize the text
    processed_text = preprocess_and_vectorize(text)
    
    # Perform prediction
    prediction = pipeline.predict([processed_text])[0]
    
    # Convert prediction to human-readable format
    category = "Fake" if prediction == 0 else "Real"
    
    return category

news = "Prime Minister Narendra Modi files nomination for Varanasi Lok Sabha seat with diverse support from astrology scholar Ganeshwar Shastri Dravid, Lalchand Kushwaha representing OBC segment, Sanjay Sonkar from Dalit community, and Baijnath Patel, reflecting broad social group backing."
result = predict_news_category(news, preprocess_and_vectorize, pipeline)
print("The news is :", result)