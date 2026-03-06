from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from math import ceil
from rapidfuzz import process
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================
# LOAD MODELS & DATA
# =========================

books = pickle.load(open('model/df.pkl', 'rb'))
df = books  # same dataset

popular_df = pickle.load(open('model/popular_df.pkl', 'rb'))
pt = pickle.load(open('model/pt.pkl', 'rb'))
similarity_score = pickle.load(open('model/similarity_score.pkl', 'rb'))

next_word = pickle.load(open('model/next_word.pkl', 'rb'))
tokenizer_next = pickle.load(open('model/tokenizer.pkl', 'rb'))

# Load pre-trained search model
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
tfidf_matrix = pickle.load(open('model/tfidf_matrix.pkl', 'rb'))

# FAST lookup table
book_lookup = books.drop_duplicates('Book-Title').set_index('Book-Title')

# =========================
# HELPER FUNCTIONS
# =========================

def next_word_predictor(text):
    try:
        token_list = tokenizer_next.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=27, padding='pre')
        predicted = next_word.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted)

        for word, index in tokenizer_next.word_index.items():
            if index == predicted_index:
                return word
        return ""
    except:
        return ""

def book_search(text, top_n=10):
    query_vec = vectorizer.transform([text])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-20:][::-1]

    titles = df.iloc[top_indices]["Book-Title"]

    # remove duplicate editions
    cleaned = []
    seen = set()

    for title in titles:
        base = title.split('(')[0].strip().lower()
        if base not in seen:
            cleaned.append(title)
            seen.add(base)
        if len(cleaned) == top_n:
            break

    return cleaned

# =========================
# ROUTES
# =========================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template(
        'index.html',
        book_name=list(popular_df['Book-Title']),
        author=list(popular_df['Book-Author']),
        image=list(popular_df['Image-URL-M']),
        votes=list(popular_df['num_ratings']),
        rating=list(popular_df['avg_rating'])
    )

@app.route('/recommend')
def recommend_ui():
    return render_template(
        'recommend.html',
        top_books_name=list(popular_df['Book-Title']),
        top_books_author=list(popular_df['Book-Author']),
        top_books_image=list(popular_df['Image-URL-M']),
        top_books_votes=list(popular_df['num_ratings']),
        top_books_rating=list(popular_df['avg_rating'])
    )

# =========================
# AUTOCOMPLETE NEXT WORD
# =========================

@app.route('/predict_word', methods=['POST'])
def predict_word():
    user_input = request.form.get('text', '').strip()
    predicted = next_word_predictor(user_input)
    return jsonify({'next_word': predicted})

# =========================
# RECOMMEND SIMILAR BOOKS
# =========================

@app.route('/recommend_books', methods=['POST'])
def recommend_books():
    user_input = request.form.get('user_input').strip()

    match, score, _ = process.extractOne(
        user_input,
        pt.index,
        score_cutoff=60
    )

    if match is None:
        return render_template('recommend.html', error="Book not found.")

    book_index = np.where(pt.index == match)[0][0]
    distances = similarity_score[book_index]

    similar_items = sorted(
        enumerate(distances),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    data = []

    for i in similar_items:
        title = pt.index[i[0]]
        if title in book_lookup.index:
            row = book_lookup.loc[title]
            data.append([title, row['Book-Author'], row['Image-URL-M']])

    return render_template('recommend.html', data=data)

# =========================
# SEARCH ROUTES
# =========================

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_text = request.form.get('search_query', '').strip()

        if not search_text:
            return render_template('recommend.html', error="Please enter a search query.")

        try:
            titles = book_search(search_text)

            results = []
            for title in titles:
                if title in book_lookup.index:
                    row = book_lookup.loc[title]
                    results.append([title, row['Book-Author'], row['Image-URL-M']])

            return render_template('recommend.html', data=results)

        except Exception as e:
            return render_template('recommend.html', error=str(e))

    return render_template('recommend.html')

@app.route('/similar/<book>')
def similar(book):
    titles = book_search(book)
    return jsonify(titles)







# =========================
# BOOK LIBRARY (PAGINATION)
# =========================

@app.route('/Book_Library')
def all_books():
    PER_PAGE = 24
    page = request.args.get('page', 1, type=int)

    books_clean = books[['Book-Title','Book-Author','Image-URL-M','Publisher']] \
                    .drop_duplicates('Book-Title')

    total_books = len(books_clean)
    total_pages = ceil(total_books / PER_PAGE)

    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE

    book_data = books_clean.iloc[start:end].values.tolist()

    start_page = max(1, page - 2)
    end_page = min(total_pages, page + 2)

    return render_template(
        'allbooks.html',
        data=book_data,
        page=page,
        total_pages=total_pages,
        start_page=start_page,
        end_page=end_page
    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)