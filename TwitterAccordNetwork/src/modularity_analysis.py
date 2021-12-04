from pathlib import Path
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import json
import csv

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

N_KEYWORDS = 10

MIN_TOPIC_PROB = 0.7

METRICS = ['pageranks', 'eigencentrality', 'betweenesscentrality', 'closnesscentrality']

if __name__ == '__main__':
    csv_path = ROOT.joinpath(Path(f"data/modularity_analysis/final_analysis_v2.csv"))
    data_path = ROOT.joinpath(Path(f"data/json/{DSET}-sentiment.json"))
    ldamodel_path = ROOT.joinpath(Path(f"models/ldamodel.npz"))
    dict_path = ROOT.joinpath(Path(f"models/corpus_dict.txt"))

    print("Loading data...")
    with data_path.open('r') as f:
        posts = json.load(f)
    
    with csv_path.open('r') as f:
        rows = [l for l in csv.DictReader(f)]
        groups = {}

        for row in rows:
            for key in row.keys() - set(['Id', 'modularity_class', 'Label']):
                try:
                    val = float(row[key])
                    row[key] = val
                except:
                    pass
            
            row['modularity_class'] = int(row['modularity_class'])
            row['data'] = posts[row['Id']]
            group_id = row['modularity_class']

            if group_id not in groups:
                groups[group_id] = {}
            
            groups[group_id][row['Id']] = row

    print("Loading LDA...")
    lda = LdaModel.load(str(ldamodel_path))
    vocab = Dictionary.load_from_text(str(dict_path))

    print("Preprocessing data...")
    for post in posts.values():
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
    
    with_topic_posts = [post for post in posts.items() if post[1]['topics']]
    print(f"Found {len(with_topic_posts)} with at least 1 topic.")

    print("Creating modularity group summaries...")
    group_summaries = {}

    for group_id, group in groups.items():
        print(f"Analyzing group {group_id}")
        max_vals = {metric: 0 for metric in METRICS}
        max_posts = {metric: [] for metric in METRICS}
        max_min_posts = {metric: [] for metric in METRICS}
        group_topics = {}

        for post in group.values():
            topics = post['data']['topics']
            min_post = {
                "text": post["data"]["full_text"],
                "clean_text": post["data"]["clean_text"],
                "id": post["Id"],
            }

            for metric in METRICS:
                if max_vals[metric] == post[metric]:
                    max_posts[metric].append(post)
                    max_min_posts[metric].append(min_post)
                elif max_vals[metric] < post[metric]:
                    max_posts[metric] = [post]
                    max_min_posts[metric] = [min_post]
                    max_vals[metric] = post[metric]
            
            for topic in topics:
                if topic not in group_topics:
                    group_topics[topic] = 0
                
                group_topics[topic] += 1
        
        group_summaries[group_id] = {
            "central_posts": max_posts,
            "central_texts": max_min_posts,
            "topics": group_topics,
            "size": len(group),
        }

    for i in range(len(group_summaries)):
        group = group_summaries[i]
        topics = group["topics"]

        max_topic_count = 0
        max_topic = 0

        for topic in topics.keys():
            if topics[topic] > max_topic_count:
                max_topic_count = topics[topic]
                max_topic = topic

        texts = group["central_texts"]
        terms = lda.get_topic_terms(max_topic, N_KEYWORDS)

        print("=" * 20 + f" GROUP {str(i)} ({group['size']}) " + "=" * 20)
        print("CENTRAL TEXTS")
        
        for metric in texts.keys():
            clean_texts = set([text["text"] for text in texts[metric]])

            print("=" * 20 + f" BY {metric} ({len(clean_texts)}) " + "=" * 20)

            for j, text in enumerate(sorted(clean_texts)):
                print("=" * 10 + f"POST {j}" + "=" * 10)
                print(text)
                print()

        print("=" * 20 + f" TOPIC {max_topic} KEYWORDS " + "=" * 20)

        for term in terms:
            print(vocab[term[0]])
        
        print()

        print("Press ENTER to continue to the next group's summary")
        input()

    print("Done!")
