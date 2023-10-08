from setuptools import setup, find_packages

setup(
    name='word2vec_similarity',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gensim',
        'numpy',
        'pandas',
        'pytest'
    ],
)
