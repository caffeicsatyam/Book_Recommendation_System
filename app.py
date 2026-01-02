from flask import Flask, render_template
import pickle
import pandas as pd

books = pickle.load(open('model/popular.pkl', 'rb'))
popular_df = pickle.load(open('model/popular_df.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():

    image_urls = []

    for m, l, s in zip(
        books['Image-URL-M'],
        books['Image-URL-L'],
        books['Image-URL-S']
    ):
        if pd.notna(m) and m != "":
            image_urls.append(m)
        elif pd.notna(l) and l != "":
            image_urls.append(l)
        else:
            image_urls.append(s)

    return render_template(
        'index.html',
        book_name=list(popular_df['Book-Title'].values),
        author=list(books['Book-Author'].values),
        image=image_urls,                       # ✅ ONE clean list
        votes=list(popular_df['num_ratings'].values),
        rating=list(popular_df['avg_rating'].values)
    )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


if __name__ == "__main__":
    app.run(debug=True)
