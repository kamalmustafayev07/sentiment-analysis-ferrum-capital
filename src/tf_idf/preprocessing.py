import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download once
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english")) - {
    "not", "no", "nor", "never", "dont", "isnt", "cant", "won't"
}
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Text preprocessing:
    - lowercase
    - remove digits
    - remove punctuation
    - remove stopwords
    - lemmatization
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(tokens)

