import string
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def case_folding(sentence):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emojis in the first range
        u"\U0001F300-\U0001F5FF"  # Emojis in the second range
        u"\U0001F680-\U0001F6FF"  # Emojis in the third range
        u"\U0001F700-\U0001F77F"  # Emojis in the fourth range
        u"\U0001F780-\U0001F7FF"  # Emojis in the fifth range
        u"\U0001F800-\U0001F8FF"  # Emojis in the sixth range
        u"\U0001F900-\U0001F9FF"  # Emojis in the seventh range
        u"\U0001FA00-\U0001FA6F"  # Emojis in the eighth range
        u"\U0001FA70-\U0001FAFF"  # Emojis in the ninth range
        u"\U0001F004-\U0001F0CF"  # Emojis in the tenth range
        "]+", flags=re.UNICODE)

    sentence = emoji_pattern.sub(r'', sentence)
    sentence = sentence.translate(str.maketrans("","", string.punctuation)).lower()
    sentence = re.sub(r"\d+", "", sentence)
    sentence = sentence.replace("/", " ")
    return sentence

def load_abbreviation_file(file_path):
    try:
        with open(file_path, "r") as file:
            abbreviations = json.load(file)
        return abbreviations
    except FileExistsError:
        print(f"File not found {file_path}")
        return {}

#Reading the abbreviation file path for preprocessing
file_path = "abbreviation_file.txt"
abbreviation_file = load_abbreviation_file(file_path)

def normalize_text(sentence):
    words = sentence.lower().split()
    words_normalized = []
    for word in words:
        for full_form, abbreviations in abbreviation_file.items():
            if word.lower() in abbreviations:
                words_normalized.append(full_form)
                break
        else:
            words_normalized.append(word)
    return " ".join(words_normalized)

def stopwords_removal(sentence):
    tokens = word_tokenize(sentence)
    liststopwords =  set(stopwords.words('indonesian'))

    custom_stopwords_file = "more_stopwords.txt"

    custom_stopwords = set()
    with open(custom_stopwords_file, "r") as file:
        for line in file:
            custom_stopwords.add(line.strip())

    combined_stopwords = liststopwords.union(custom_stopwords)

    with open(custom_stopwords_file, "w") as file:
        for word in combined_stopwords:
            file.write(word + "\n")

def remove_custom_stopwords(sentence, custom_stopwords_file):
    custom_stopwords = set()
    with open(custom_stopwords_file, 'r') as file:
        for line in file:
            custom_stopwords.add(line.strip())

    words = word_tokenize(sentence)

    filtered_words = [word for word in words if word.lower() not in custom_stopwords]

    cleaned_text = ' '.join(filtered_words)

    return cleaned_text

def stemming_text(sentence):
    factory = StemmerFactory()
    Stemmer = factory.create_stemmer()

    sentence = Stemmer.stem(sentence)
    return sentence