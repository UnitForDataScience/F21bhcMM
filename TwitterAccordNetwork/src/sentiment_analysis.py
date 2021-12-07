from transformers import pipeline
from pathlib import Path
from tqdm.auto import tqdm
import json

# Dataset of ~15,000 English COVID-19 Tweets from 3-4PM on January 6th, Wednesday, 2021
DSET = "3pm-eng-dataset"

# Path to root directory of project
ROOT = Path(__file__).parent.parent

# Remove ALL non-alphanumeric characters, retaining monospacing
def to_alnum(text):
    return " ".join(filter(None, ["".join(filter(str.isalnum, word)) for word in text.split(" ")]))

if __name__ == "__main__":
    data_path = ROOT.joinpath(Path(f"data/json/{DSET}-cleaned.json"))
    out_path = ROOT.joinpath(Path(f"data/json/{DSET}-sentiment.json"))

    print("Loading data...")
    with data_path.open('r') as f:
        posts = json.load(f)
    
    posts = {k: v for k, v in posts.items() if v}

    print("Analyzing sentiments...")
    model = pipeline("sentiment-analysis")
    
    for post in tqdm(posts.values()):
        post['sentiment'] = model(to_alnum(post['clean_text']))[0]
    
    print("Saving...")

    with out_path.open('w') as f:
        json.dump(posts, f)

    print("Finished!")
