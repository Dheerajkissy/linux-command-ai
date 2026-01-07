import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import nltk

# Download stopwords if not present
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and normalizes input text for NLP processing
    """
    # Convert to lowercase
    text = text.lower()

    # Tokenize text into words (no punkt dependency)
    tokens = wordpunct_tokenize(text)

    # Remove stopwords and punctuation
    tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return " ".join(tokens)
