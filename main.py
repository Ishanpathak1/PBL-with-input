import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter

text=input("Enter Your text Here: ")

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Tokenize each sentence into words
words = [word_tokenize(sent) for sent in sentences]
words = [word for sent in words for word in sent]

# Remove stopwords and lowercase the text
stop_words = set(stopwords.words("english"))
words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
print(words)

# Create a bag of words
bag_of_words = Counter(words)

# Calculate the frequency of each word
total_words = sum(bag_of_words.values())
frequency = {word: count/total_words for word, count in bag_of_words.items()}
print(frequency)

# Calculate the term frequency
tf = {word: count/total_words for word, count in bag_of_words.items()}

print(tf)