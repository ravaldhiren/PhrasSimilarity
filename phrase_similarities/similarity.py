import gensim
from gensim.models import KeyedVectors
import numpy as np


class Word2VecSimilarity:
    def __init__(self, word2vec_model_path,model_encoding='unicode_escape'):
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path)

    def calculate_l2_distance(self, phrase1, phrase2):
        phrase1_vector = np.sum([self.get_word_embedding(word) for word in phrase1.split()])
        phrase2_vector = np.sum([self.get_word_embedding(word) for word in phrase2.split()])

        return np.linalg.norm(phrase1_vector - phrase2_vector)

    def calculate_similarity_matrix(self, phrases):
        similarity_matrix = np.zeros((len(phrases), len(phrases)))

        for i in range(len(phrases)):
            for j in range(len(phrases)):
                if i != j:
                    similarity_matrix[i][j] = self.calculate_l2_distance(phrases[i], phrases[j])

        return similarity_matrix

    def find_closest_match(self, phrase, phrases):
        closest_match = None
        closest_match_distance = np.inf

        for p in phrases:
            distance = self.calculate_l2_distance(phrase, p)
            if distance < closest_match_distance:
                closest_match = p
                closest_match_distance = distance

        return closest_match, closest_match_distance

    def get_word_embedding(self, word):
        try:
            return self.word2vec_model.get_vector(word)
        except KeyError:
            return self.word2vec_model.get_vector("how")
