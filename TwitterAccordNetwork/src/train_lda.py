from pathlib import Path
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import numpy as np
import json

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

N_TOPICS = 100

if __name__ == '__main__':
    data_path = ROOT.joinpath(Path(f"data/json/{DSET}-cleaned.json"))
    dict_path = ROOT.joinpath(Path(f"models/corpus_dict.txt"))
    model_path = ROOT.joinpath(Path(f"models/ldamodel.npz"))

    print("Loading data...")
    with data_path.open('r') as f:
        posts = json.load(f)
    
    texts = [post['clean_text'] for post in posts.values()]
    processed_texts = preprocess_documents(texts)
    vocab = corpora.Dictionary(processed_texts)
    vocab.save_as_text(str(dict_path))
    corpus = [vocab.doc2bow(l) for l in processed_texts]

    print("Training model...")
    model = LdaModel(corpus, N_TOPICS, vocab)

    print("Calculating average topic coherence (C_V)...")
    top = model.top_topics(corpus, processed_texts, vocab, coherence='c_v')

    print(f"Coherence: {np.mean([t[1] for t in top])}")

    print("Saving model...")
    model.save(str(model_path))

    print("Finished!")
