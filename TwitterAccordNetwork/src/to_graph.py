from pathlib import Path
from gensim.models.word2vec import Word2Vec
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine
import networkx as nk
import numpy as np
import json

from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_punctuation, remove_stopwords, preprocess_string

CUSTOM_FILTERS = [
    remove_stopwords,
    strip_punctuation,
    strip_multiple_whitespaces,
]

# Dataset of ~15,000 English COVID-19 Tweets from 3-4PM on January 6th, Wednesday, 2021
DSET = "3pm-eng-dataset"

# Path to root directory of project
ROOT = Path(__file__).parent.parent

SIMILARITY_THRESH = 0.999
N_KEYWORDS = 10
WEIGHTED = True

MIN_TOPIC_PROB = 0.7

if __name__ == '__main__':
    data_path = ROOT.joinpath(Path(f"data/json/{DSET}-sentiment.json"))
    w2vmodel_path = ROOT.joinpath(Path(f"models/w2vmodel.npz"))
    ldamodel_path = ROOT.joinpath(Path(f"models/ldamodel.npz"))
    dict_path = ROOT.joinpath(Path(f"models/corpus_dict.txt"))

    print("Loading data...")
    with data_path.open('r') as f:
        posts = list(json.load(f).values())

    print("Loading LDA...")
    lda = LdaModel.load(str(ldamodel_path))
    vocab = Dictionary.load_from_text(str(dict_path))

    print("Loading W2V...")
    wv = Word2Vec.load(str(w2vmodel_path)).wv

    print("Preprocessing data...")
    for post in posts:
        post['sentiment_score'] = post['sentiment']['score'] * (-1 if post['sentiment']['label'] == 'NEGATIVE' else 1)
        bow = vocab.doc2bow(preprocess_string(post['clean_text'], filters=CUSTOM_FILTERS))
        topics = lda.get_document_topics(bow, MIN_TOPIC_PROB)

        # Only assign singular max probability topic to posts
        max_p = 0
        max_t = -1

        for topic in topics:
            if topic[1] > max_p:
                max_p = topic[1]
                max_t = topic[0]

        if max_t != -1:
            post['topics'] = [max_t]
        else:
            post['topics'] = []
    
    with_topic_posts = [post for post in posts if post['topics']]
    print(f"Found {len(with_topic_posts)} with at least 1 topic.")

    print("Calculating topic similarities...")
    n_topics = len(lda.get_topics())
    topic_similarity_matrix = np.zeros((n_topics, n_topics), dtype=bool)
    np.fill_diagonal(topic_similarity_matrix, True)

    for i in tqdm(range(n_topics - 1)):
        topic1 = lda.get_topic_terms(i, N_KEYWORDS)
        topic1_terms = [wv[vocab[word[0]]] * (word[1] if WEIGHTED else 1) for word in topic1]
        topic1_vector = np.sum(topic1_terms, axis=0)

        for j in range(i + 1, n_topics):
            topic2 = lda.get_topic_terms(j, N_KEYWORDS)
            topic2_terms = [wv[vocab[word[0]]] * (word[1] if WEIGHTED else 1) for word in topic2]
            topic2_vector = np.sum(topic2_terms, axis=0)
            similarity = 1 - cosine(topic1_vector, topic2_vector)

            if similarity >= SIMILARITY_THRESH:
                topic_similarity_matrix[i][j] = True
                topic_similarity_matrix[j][i] = True
    
    print("Gathering weights...")
    for i in tqdm(range(len(with_topic_posts) - 1)):
        p1 = with_topic_posts[i]
        p1["agree_weights"] = {}
        p1["disagree_weights"] = {}
        t1s = p1["topics"]
        w1 = p1["sentiment_score"]

        for j in range(i + 1, len(with_topic_posts)):
            p2 = with_topic_posts[j]
            p2id = p2["id"]
            t2s = p2["topics"]
            w2 = p2["sentiment_score"]

            p1["agree_weights"][p2id] = 0
            p1["disagree_weights"][p2id] = 0

            for t1 in t1s:
                for t2 in t2s:
                    if topic_similarity_matrix[t1][t2]:
                        if (w1 < 0 and w2 < 0) or (w1 > 0 and w2 > 0):
                            agree_weight = abs(w1 + w2) / 2.0
                            p1["agree_weights"][p2id] += agree_weight
                        else:
                            disagree_weight = abs(w1 - w2) / 2.0
                            p1["disagree_weights"][p2id] += disagree_weight

    print("Constructing graphs...")
    agree_graph = nk.Graph()
    disagree_graph = nk.Graph()
    
    print("Adding nodes...")
    for post in with_topic_posts:
        agree_graph.add_node(post["id"])
        disagree_graph.add_node(post["id"])

    print("Adding weights...")
    for post in tqdm(with_topic_posts[:-1]):
        for post_id, weight in post["agree_weights"].items():
            if weight > 0:
                agree_graph.add_edge(post["id"], post_id, weight=weight)
        
        for post_id, weight in post["disagree_weights"].items():
            if weight > 0:
                disagree_graph.add_edge(post["id"], post_id, weight=weight)

    print("Saving...")
    agree_path = ROOT.joinpath(Path("data/graphs/agree_graph.gexf"))
    disagree_path = ROOT.joinpath(Path("data/graphs/disagree_graph.gexf"))
    nk.write_gexf(agree_graph, agree_path)
    nk.write_gexf(disagree_graph, disagree_path)

    print("Done!")
