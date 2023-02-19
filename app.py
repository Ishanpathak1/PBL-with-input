from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Read the CSV file and extract the text data
        df = pd.read_csv("IMDB_dataset.csv")
        text = " ".join(df['review'].tolist())
        # Tokenize words
        tokenized_words = word_tokenize(text)
        # Remove stop words
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in tokenized_words if word.lower() not in stop_words]
        # Count bag of words and term frequency
        bag_of_words = Counter(filtered_words)
        total_words = sum(bag_of_words.values())
        term_frequency = {word: count / total_words for word, count in bag_of_words.items()}
        # Count sentences
        sentence_count = len(sent_tokenize(text))
        return render_template("index.html", tokenized_words=tokenized_words, bag_of_words=bag_of_words, term_frequency=term_frequency, sentence_count=sentence_count)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)











