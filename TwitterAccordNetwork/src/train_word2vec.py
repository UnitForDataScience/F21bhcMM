'''
Training script for custom Word2Vec model on Tweets
'''
import json

from pathlib import Path
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_punctuation, remove_stopwords, preprocess_string

def preprocess_documents(texts):
    return [preprocess_string(t, filters=CUSTOM_FILTERS) for t in texts]

CUSTOM_FILTERS = [
    remove_stopwords,
    strip_punctuation,
    strip_multiple_whitespaces,
]

# Dataset of ~15,000 English COVID-19 Tweets from 3-4PM on January 6th, Wednesday, 2021
DSET = "3pm-eng-dataset"

# Path to root directory of project
ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    data_path = ROOT.joinpath(Path(f"data/json/{DSET}-cleaned.json"))
    model_path = ROOT.joinpath(Path(f"models/w2vmodel.npz"))

    print("Loading file...")
    with open(data_path, 'r') as f:
        posts = json.load(f)

    texts = [post['clean_text'] for post in posts.values()]

    print("Building corpus...")
    corpus = preprocess_documents(texts)

    import code
    code.interact(local=locals())

    print("Training model...")
    model = Word2Vec(corpus, min_count=1, workers=4)
    model.save(str(model_path))
    print("Finished training!")
