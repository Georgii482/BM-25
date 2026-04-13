import math
import string
from nltk.stem import WordNetLemmatizer 
from deep_translator import GoogleTranslator

print("Enter the first document:")
document_A = input().split()
print("Enter the second document:")
document_B = input().split()
print("Enter the key words:")
key_words = input().split()
punct_table = str.maketrans('', '', string.punctuation)
translator = GoogleTranslator(source='auto', target='en')
lemmatizer = WordNetLemmatizer()

def normalize(word):
    cleaned_word = word.translate(punct_table)
    translated = translator.translate(cleaned_word)
    word = lemmatizer.lemmatize(translated, "v").lower()
    return word

document_A = [normalize(word) for word in document_A]
document_B = [normalize(word) for word in document_B]
key_words = [normalize(word) for word in key_words]

avgdl = (len(document_A) + len(document_B)) / 2
N = 2
k1 = 1.2
b = 0.75
sum_BM25_A = 0
sum_BM25_B = 0

for word in key_words:

    f_A = document_A.count(word)
    f_B = document_B.count(word)
    
    if f_A > 0 and f_B > 0:
        n = 2
    elif f_A > 0 or f_B > 0:
        n = 1
    else:
        n = 0

    TF_A = (f_A * (k1 + 1)) / (f_A + k1 * (1 - b + b * (len(document_A) / avgdl)))
    TF_B = (f_B * (k1 + 1)) / (f_B + k1 * (1 - b + b * (len(document_B) / avgdl)))
    
    IDF = math.log((N - n + 0.5) / (n + 0.5)) + 1

    sum_BM25_A += TF_A * IDF
    sum_BM25_B += TF_B * IDF

print("BM25 score for Document A:", sum_BM25_A)
print("BM25 score for Document B:", sum_BM25_B)