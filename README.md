# 📚 Book Recommendation System

A machine learning-powered Flask web application that provides intelligent book recommendations based on user preferences and collaborative filtering algorithms.

## 🌟 Features

- **Smart Book Recommendations**: Get personalized book recommendations using collaborative filtering and content-based filtering
- **Book Search**: Search for books using fuzzy matching and TF-IDF vectorization
- **Next Word Prediction**: AI-powered autocomplete suggesting the next word as you type
- **Popular Books Display**: Browse the most highly-rated and reviewed books
- **Book Library**: Browse the entire book collection with pagination
- **Responsive UI**: Clean, user-friendly interface built with HTML/CSS

## 🛠️ Technology Stack

- **Backend**: Flask 3.1.3
- **Machine Learning**: TensorFlow/Keras, Scikit-learn, Pandas, NumPy
- **Search & Matching**: RapidFuzz, TF-IDF Vectorizer
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Python Version**: 3.10+

## 📋 Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## 🚀 Installation

### 1. Clone or navigate to the project directory
```bash
cd Book_Recommendation_System
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv310
.\venv310\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv310
venv310\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv310
source venv310/bin/activate
```

### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

## 📌 Project Structure

```
Book_Recommendation_System/
├── app.py                          # Main Flask application
├── test.py                         # Test file
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment configuration
├── README.md                       # Project documentation
├── book_recommendation_System.ipynb # Jupyter notebook with models
│
├── model/                          # Pre-trained models and data
│   ├── df.pkl                     # Main dataset
│   ├── popular_df.pkl             # Popular books data
│   ├── pt.pkl                     # Pivot table for recommendations
│   ├── similarity_score.pkl       # Cosine similarity scores
│   ├── vectorizer.pkl             # TF-IDF vectorizer for search
│   ├── tfidf_matrix.pkl           # TF-IDF matrix for search
│   ├── next_word.pkl              # Next word prediction model
│   └── tokenizer.pkl              # Tokenizer for predictions
│
├── templates/                      # HTML templates
│   ├── home.html                  # Home page
│   ├── index.html                 # Popular books page
│   ├── recommend.html             # Book recommendations page
│   ├── allbooks.html              # Book library page
│   ├── home_styles.css            # Home page styles
│   ├── recommend.css              # Recommendation page styles
│   └── library.css                # Library page styles
│
├── Notebook/                       # Jupyter notebooks
│   └── Book_BAY.ipynb             # Model training notebook
│
└── venv310/                        # Virtual environment (auto-generated)
```

## 🎯 Usage

### Running the Application

1. **Activate the virtual environment** (if not already activated)
```bash
# Windows PowerShell
.\venv310\Scripts\Activate.ps1

# Windows Command Prompt
venv310\Scripts\activate.bat

# macOS/Linux
source venv310/bin/activate
```

2. **Run the Flask application**
```bash
python app.py
```

3. **Open your browser** and navigate to:
```
http://localhost:5000
```

### Application Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page |
| `/index` | GET | Popular books page |
| `/recommend` | GET | Book recommendations interface |
| `/recommend_books` | POST | Get recommendations for a selected book |
| `/search` | GET, POST | Search for books |
| `/predict_word` | POST | Get next word prediction |
| `/similar/<book>` | GET | Get similar books (JSON) |
| `/Book_Library` | GET | Browse entire book collection with pagination |

## 🤖 How It Works

### Book Recommendations
The system uses two main approaches:

1. **Collaborative Filtering**: Finds books similar to a given book using cosine similarity on a book-feature matrix
2. **Content-Based Filtering**: Uses TF-IDF vectorization to search and find semantically similar books

### Next Word Prediction
Uses a pre-trained LSTM neural network to predict the next word as users type, enhancing the search experience.

### Search Functionality
- Implements fuzzy matching using RapidFuzz for intelligent book title matching
- Uses TF-IDF vectorization combined with cosine similarity for finding similar books
- Automatically removes duplicate book editions from results

## 📊 Key Features in Detail

### Popular Books
Displays the most popular and highest-rated books from the dataset to help users discover trending books.

### Book Recommendations
1. Enter a book title
2. System finds the closest match using fuzzy matching
3. Returns 5 most similar books based on user ratings patterns

### Advanced Search
- Free-text search across the entire book collection
- Intelligent deduplication of book editions
- Top 10 most relevant results

### Book Library
- Browse all available books
- Pagination support (24 books per page)
- Displays book title, author, cover image, and publisher

## 🔧 Dependencies

Key dependencies:
- Flask - Web framework
- TensorFlow/Keras - Deep learning
- Scikit-learn - Machine learning algorithms
- Pandas - Data manipulation
- NumPy - Numerical computing
- RapidFuzz - String matching
- Joblib - Model persistence

See `requirements.txt` for the complete list.

## 📝 Model Training

The pre-trained models were generated using Jupyter notebooks:
- `book_recommendation_System.ipynb` - Main model development
- `Notebook/Book_BAY.ipynb` - Additional model training

To retrain or modify models, open these notebooks with Jupyter:
```bash
jupyter notebook book_recommendation_System.ipynb
```

## 📈 Performance Optimization

- **Fast Lookup Table**: Uses indexed dataset for O(1) book lookups
- **Pre-computed Similarity**: Similarity scores are pre-computed and cached
- **Pagination**: Implements pagination for the book library to handle large datasets
- **Duplicate Removal**: Smart deduplication of book editions in search results

## 🐛 Troubleshooting

**Issue**: Models not found error
- **Solution**: Ensure the `model/` directory exists and contains all `.pkl` files
pip install --only-binary :all: scipy
**Issue**: ModuleNotFoundError
- **Solution**: Verify all packages are installed: `pip install -r requirements.txt`

**Issue**: Port already in use
- **Solution**: Change the port in `app.py`: `app.run(debug=True, port=5001)`

## 📄 License

This project is available for personal and educational use.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📧 Contact

For questions or suggestions, please open an issue in the project repository.

---

**Last Updated**: March 2026  
**Version**: 1.0.0