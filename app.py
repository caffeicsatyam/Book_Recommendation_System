import os
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(path):
    return pickle.load(open(os.path.join(BASE_DIR, path), "rb"))

try:
    books = load_model("model/df.pkl")
    popular_df = load_model("model/popular_df.pkl")
    pt = load_model("model/pt.pkl")
    similarity_score = load_model("model/similarity_score.pkl")
    vectorizer = load_model("model/vectorizer.pkl")
    tfidf_matrix = load_model("model/tfidf_matrix.pkl")
except FileNotFoundError as e:
    print(f"Error: Required model file not found: {e}")
    exit(1)

# FAST lookup table
book_lookup = books.drop_duplicates('Book-Title').set_index('Book-Title')

# =========================
# HELPER FUNCTIONS
# =========================

def get_book_suggestions(query, top_n=5):
    if not query:
        return []
    
    # Filter books containing query (case-insensitive)
    suggestions = books[books['Book-Title'].str.lower().str.contains(query.lower(), na=False)]['Book-Title'].unique()
    return suggestions[:top_n].tolist()

def book_search(text, top_n=10):
    query_vec = vectorizer.transform([text])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-20:][::-1]

    titles = books.iloc[top_indices]["Book-Title"]

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
# AUTOCOMPLETE / SUGGESTIONS
# =========================

@app.route('/suggest_titles', methods=['POST'])
def suggest_titles():
    user_input = request.form.get('query', '').strip()
    suggestions = get_book_suggestions(user_input)
    return jsonify({'suggestions': suggestions})

@app.route('/predict_word', methods=['POST'])
def predict_word():
    # Legacy endpoint retained for existing JS
    user_input = request.form.get('text', '').strip()
    suggestions = get_book_suggestions(user_input)
    predicted = suggestions[0] if suggestions else ""
    return jsonify({'next_word': predicted, 'suggestions': suggestions})

# =========================
# SEARCH & RECOMMENDATION LOGIC
# =========================

@app.route('/search', methods=['GET', 'POST'])
@app.route('/recommend_books', methods=['POST'])
def search():
    if request.method == 'POST':
        user_input = request.form.get('search_query') or request.form.get('user_input')
        if not user_input:
            return render_template('recommend.html', error="Please enter a book title.")
        
        user_input = user_input.strip()
        
        # 1. Try Similarity Recommendation first
        result = process.extractOne(user_input, pt.index, score_cutoff=70)
        
        if result:
            match, score, _ = result
            book_index = np.where(pt.index == match)[0][0]
            distances = similarity_score[book_index]
            similar_items = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:12]
            
            data = []
            # Include the matched book itself
            if match in book_lookup.index:
                row = book_lookup.loc[match]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                data.append([match, row['Book-Author'], row['Image-URL-M'], "MATCHED"])

            for i in similar_items:
                title = pt.index[i[0]]
                if title in book_lookup.index:
                    row = book_lookup.loc[title]
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    # Avoid duplicated title in result
                    if title != match:
                        data.append([title, row['Book-Author'], row['Image-URL-M'], "RECOMMENDED"])
            
            return render_template('recommend.html', data=data, query=user_input)

        # 2. Fallback to TF-IDF Search
        try:
            titles = book_search(user_input)
            results = []
            for title in titles:
                if title in book_lookup.index:
                    row = book_lookup.loc[title]
                    if isinstance(row, pd.DataFrame): row = row.iloc[0]
                    results.append([title, row['Book-Author'], row['Image-URL-M'], "SEARCH RESULTS"])

            if not results:
                return render_template('recommend.html', error=f"No books found for '{user_input}'.", query=user_input)

            return render_template('recommend.html', data=results, query=user_input)

        except Exception as e:
            return render_template('recommend.html', error=str(e), query=user_input)

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
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"Server started on port {port}")
    serve(app, host='0.0.0.0', port=port)