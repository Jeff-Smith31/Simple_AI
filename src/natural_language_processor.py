import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# Sample text for analysis
text = """
Natural Language Processing (NLP) is a branch of artificial intelligence that helps 
computers understand, interpret, and manipulate human language. NLP combines computational 
linguistics with statistical, machine learning, and deep learning models. The applications 
of NLP include sentiment analysis, speech recognition, and machine translation.
"""


def analyze_text(text):
    # 1. Basic text statistics
    print("=== Basic Text Analysis ===")
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    print(f"Number of sentences: {len(sentences)}")
    print(f"Number of words: {len(words)}")

    # 2. Tokenization and Stop Words Removal
    print("\n=== Tokenization and Stop Words ===")
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words
                      and word.isalnum()]
    print(f"Words after removing stop words: {filtered_words[:10]}...")

    # 3. Part of Speech Tagging
    print("\n=== Parts of Speech Tagging ===")
    pos_tags = nltk.pos_tag(words)
    print("First 10 words with POS tags:", pos_tags[:10])

    # 4. Lemmatization
    print("\n=== Lemmatization ===")
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    print(f"Lemmatized words: {lemmatized_words[:10]}...")

    # 5. Word Frequency Analysis
    print("\n=== Word Frequency Analysis ===")
    word_freq = Counter(lemmatized_words)
    print("Most common words:")
    for word, count in word_freq.most_common(5):
        print(f"{word}: {count}")

    # 6. Using spaCy for advanced NLP
    print("\n=== SpaCy Analysis ===")
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    # Named Entity Recognition
    print("\nNamed Entities:")
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_}")

    # Dependency Parsing
    print("\nDependency Parsing (first sentence):")
    for token in list(doc.sents)[0]:
        print(f"{token.text}: {token.dep_}")


def train_text_classifier(texts, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    classifier = MultinomialNB()
    classifier.fit(X, labels)
    return vectorizer, classifier


def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


# Run the analysis
if __name__ == "__main__":
    analyze_text(text)
#    train_text_classifier(text, labels)
    analyze_sentiment(text)
