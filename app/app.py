import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from scipy.sparse import csr_matrix
import json
from flask import Flask, request, jsonify, render_template

data_path = "data"
nlp = spacy.load("en_core_web_sm")  

df = pd.read_csv("data/IMDB Dataset.csv", sep=',', encoding='utf-8')

def processing_data(dataframe):
    clean_data = []
    nltk.download('stopwords')
    nltk.download('wordnet')
    for review in df['review']: 
        soup = BeautifulSoup(review, 'html.parser') 
        text = soup.get_text()   
        clean_data.append(text)

    reviews = [review.lower() for review in clean_data]
    # DEFINE STOPWORDS
    stop_words = stopwords.words('english')

    clean_data = []

    # ITERATE OVER LIST OF STRINGS TO REMOVE A STOPWORDS
    for review in reviews:

        # GET ALL WORD THAT ARE NOT A STOPWORDS
        clean_text = [word for word in review.split() if word not in stop_words]
        clean_text = ' '.join(clean_text)   # COMBINES EVERY WORD INTO SENTENCES

        clean_data.append(clean_text)   # PUSH EVERY REVIEW INTO LIST

    lemma = WordNetLemmatizer()

    lemmatized_data = []

    # ITERATE OVER LIST OF STRINGS TO LEMMATIZE 
    for review in clean_data:

        # LEMMATIZE ADJECTIVES TOKEN/WORD
        clean_text_1 = [lemma.lemmatize(word= word, pos='a') for word in review.split()]  
        clean_text_1 = ' '.join(clean_text_1)  # COMBINES EVERY WORD INTO SENTENCES

        # LEMMATIZE VERB TOKEN/WORD
        clean_text_2 = [lemma.lemmatize(word = word, pos='v') for word in clean_text_1.split()] 
        clean_text_2 = ' '.join(clean_text_2) # COMBINES EVERY WORD INTO SENTENCES

        # LEMMATIZE NOUN TOKEN/WORD
        clean_text_3 = [lemma.lemmatize(word= word, pos='n') for word in clean_text_2.split()] 
        clean_text_3 = ' '.join(clean_text_3)

        lemmatized_data.append(clean_text_3) 


    final_data = []

    # ITERATE OVER LIST OF STRINGS TO TOKENIZE EVERY REVIEW
    for review in lemmatized_data:
        text = [review.split()]
        final_data.append(text)
    
    return final_data

if not os.path.isfile(os.path.join(data_path, 'data_processed.csv')):
    data_processed = processing_data(df)
    data_processed = pd.DataFrame(data_processed)
    data_processed.to_csv(os.path.join(data_path, 'data_processed.csv'))
else:
    data_processed = pd.read_csv(os.path.join(data_path, 'data_processed.csv'))



vectorizer = TfidfVectorizer(stop_words='english',max_features=5000)

# Huấn luyện và chuyển đổi văn bản thành ma trận TF-IDF

tfidf_matrix = vectorizer.fit_transform(data_processed["0"])
# Giả sử tfidf_matrix là đầu ra từ TfidfVectorizer
if isinstance(tfidf_matrix, csr_matrix):  # Nếu là sparse matrix
    tfidf_matrix = tfidf_matrix.toarray()  # Chuyển về dense
    tfidf_matrix = tfidf_matrix.astype('float32')  # Đảm bảo float32

print(tfidf_matrix.shape)
# Tạo index Flat sử dụng khoảng cách L2
index = faiss.IndexFlatL2(5000)
index.add(tfidf_matrix)

def get_query(string):
    query = vectorizer.transform(string.split()).toarray().astype('float32')

    distances, indices = index.search(query, 5)
    result = []
    pd.options.display.max_colwidth = 100000
    for x in indices:
        result.append(str(df['review'][x]))
    return result

# Tạo đối tượng Flask
app = Flask(__name__)

# Định nghĩa route cho trang chủ
@app.route('/')
def home():
    return render_template('index.html', name="Học viên")
    # return "Chào mừng bạn đến với Flask!"

@app.route('/form', methods=['GET', 'POST'])
def form_example():
    return render_template('form.html')

@app.route('/handle_get', methods=['GET'])
def handle_get():
    if request.method == 'GET':
        query = request.args['query']

    return render_template('result.html', result = get_query(query))
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)