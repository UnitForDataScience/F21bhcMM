from pathlib import Path
import re
import json

from gensim.parsing.preprocessing import strip_multiple_whitespaces, preprocess_string

PUNCTS = [
    '.', ',', '#', '-', '@', ':', ';', '?', '!', '_', '"', '(', ')', '[', ']', '{', '}', "&amp;", '\\', '/'
]

EMOJI_PATTERN = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def remove_links(text):
    return " ".join(filter(lambda x: not x.startswith("https"), text.split(" ")))

def remove_punctuation(text: str, puncts=PUNCTS):
    for punct in puncts:
        text = text.replace(punct, " ")

    return text

def preprocess_documents(texts):
    return [preprocess_string(t, filters=CUSTOM_FILTERS) for t in texts]

CUSTOM_FILTERS = [
    lambda x: x.lower(),
    lambda x: EMOJI_PATTERN.sub(r' ', x),
    strip_multiple_whitespaces,
    remove_links,
    remove_punctuation,
    strip_multiple_whitespaces,
]

# Dataset of ~15,000 English COVID-19 Tweets from 3-4PM on January 6th, Wednesday, 2021
DSET = "3pm-eng-dataset"

# Path to root directory of project
ROOT = Path(__file__).parent.parent

if __name__ == '__main__':
    data_path = ROOT.joinpath(Path(f"data/json/{DSET}.json"))
    out_path = ROOT.joinpath(Path(f"data/json/{DSET}-cleaned.json"))

    print("Loading data...")
    with data_path.open('r') as f:
        posts = json.load(f)
    
    print("Cleaning data...")
    posts = {k: v for k, v in posts.items() if v}

    for post in posts.values():
        post['clean_text'] = " ".join(preprocess_string(post['full_text'], filters=CUSTOM_FILTERS))

    print("Saving cleaned data...")
    with out_path.open('w') as f:
        json.dump(posts, f)
    
    print("Finished!")
