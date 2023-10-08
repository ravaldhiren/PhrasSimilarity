
import logging
import pandas as pd
from similarity import Word2VecSimilarity
import util


def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Load the phrases.csv file
    phrases = util.load_phrases_csv('../data/phrases.csv')

    # # Load the pre-trained Word2Vec model
    #util.generate_vector_csv(util.get_vectors_path())
    word2vec_similarity = Word2VecSimilarity(util.get_vectors_path())

    # Calculate the similarity matrix
    similarity_matrix = word2vec_similarity.calculate_similarity_matrix(phrases)
    logging.info(similarity_matrix)

    # Find the closest match to a user-input phrase
    user_input_phrase = 'What were the premiums earned by the industry in 2030?'
    closest_match, closest_match_distance = word2vec_similarity.find_closest_match(user_input_phrase, phrases)
    logging.info(
        f'The closest match to "{user_input_phrase}" is "{closest_match}" with a distance of {closest_match_distance}')


if __name__ == '__main__':
    main()
