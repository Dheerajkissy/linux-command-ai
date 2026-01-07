import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text

# Load dataset
data = pd.read_csv("data/commands.csv")

# Preprocess command descriptions
data["processed_desc"] = data["description"].apply(preprocess_text)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Convert descriptions to TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(data["processed_desc"])

def recommend_command(user_input):
    """
    Recommends the most relevant Linux command
    based on user input using cosine similarity
    """
    # Preprocess user input
    user_input_processed = preprocess_text(user_input)

    # Vectorize user input
    user_vector = vectorizer.transform([user_input_processed])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

    # Get index of best match
    best_match_index = similarity_scores.argmax()

    # Return corresponding command
    return data.iloc[best_match_index]["command"]

if __name__ == "__main__":
    user_query = input("Describe your task: ")
    command = recommend_command(user_query)
    print(f"\nRecommended Command: {command}")
