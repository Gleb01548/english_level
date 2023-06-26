import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt")
stop_words = stopwords.words("english")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocessing_text(x, use_stop_words: bool = True, type_processing=False):
    if use_stop_words:
        x = [word for word in word_tokenize(x) if word not in stop_words]
        x = " ".join(x)

    if type_processing == "stemmer":
        x = [stemmer.stem(word) for word in word_tokenize(x)]
        x = " ".join(x)

    if lemmatizer == "lemm":
        x = [lemmatizer.lemmatize(word) for word in word_tokenize(x)]
        x = " ".join(x)
    return x
