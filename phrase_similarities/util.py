import pandas as pd
import gensim
from gensim.models import KeyedVectors


def load_phrases_csv(file_path):
    return pd.read_csv(file_path, encoding='unicode_escape')['Phrases'].tolist()


def generate_vector_csv(file_path):
    location = "../data/GoogleNews-vectors-negative300.bin"
    wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=100000)
    wv.save_word2vec_format(get_vectors_path())


def get_vectors_path():
    return '../data/vectors.csv'
