from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from math import ceil

# Load the models and data
books = pickle.load(open('model/popular.pkl', 'rb'))
popular_df = pickle.load(open('model/popular_df.pkl', 'rb'))
pt = pickle.load(open('model/pt.pkl', 'rb'))
similarity_score = pickle.load(open('model/similarity_score.pkl', 'rb'))

app = Flask(__name__)

# This handles the landing page and the /Home route
@app.route('/')
def home():
    # This renders the new minimalist landing page
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template(
        'index.html',
        book_name = list(popular_df['Book-Title'].values),
        author = list(popular_df['Book-Author'].values),
        image = list(popular_df['Image-URL-M'].values),
        votes = list(popular_df['num_ratings'].values),
        rating = list(popular_df['avg_rating'].values)
    )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend_books():
    user_input = request.form.get('user_input').strip()

    # Match check
    if user_input not in pt.index:
        return render_template('recommend.html', error="Book not found.")

    index = np.where(pt.index == user_input)[0][0]
    distances = similarity_score[index]
    similar_items = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        item.extend(list(temp_df['Book-Title'].values))
        item.extend(list(temp_df['Book-Author'].values))
        item.extend(list(temp_df['Image-URL-M'].values))
        data.append(item)

    return render_template('recommend.html', data=data)

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

    # ✅ pagination window logic (HERE, not in Jinja)
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

if __name__ == "__main__":
    app.run(debug=True)