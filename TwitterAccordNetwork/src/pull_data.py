'''
This script pulls data from the Twitter API and writes it into a JSON file
Dataset of Twitter IDs should be provided in a tsv file in data/raw
'''
from dotenv import load_dotenv
from pathlib import Path
from tqdm.auto import tqdm
import tweepy
import os
import json
import csv

# Dataset of ~15,000 English COVID-19 Tweets from 3-4PM on January 6th, Wednesday, 2021
DSET = "3pm-eng-dataset"

# Path to root directory of project
ROOT = Path(__file__).parent.parent

if __name__ == '__main__':
    # Load env file
    env_path = ROOT.joinpath(".env")
    load_dotenv(env_path)

    # Read Tweet/status IDs from specified dataset
    csv_path = ROOT.joinpath(Path(f"data/raw/{DSET}.tsv"))

    with csv_path.open('r') as f:
        reader = csv.reader(f, delimiter='\t')
        status_ids = [row[0] for row in list(reader)[1:]]

    # Create tweepy API instance with env keys
    consumer_key = os.environ.get("CONSUMER_KEY")
    consumer_secret = os.environ.get("CONSUMER_SECRET")
    access_token = os.environ.get("ACCESS_TOKEN")
    access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    api.verify_credentials()

    # Rate limited to 900 / 15 minutes, calculate quick time estimate
    rate_lim_mins = len(status_ids) // 900 * 15
    print(f"Pulling all {len(status_ids)} tweets will result in rate limits of {rate_lim_mins // 60} hours and {rate_lim_mins % 60} minutes.")

    # Read all IDs into JSON-serializable dictionary
    results = {}
    broken = 0

    for status_id in tqdm(status_ids):
        try:
            response = api.get_status(status_id, tweet_mode="extended")
            results[status_id] = response._json
            broken = 0
        except KeyboardInterrupt:
            break
        except:
            results[status_id] = {}
            broken += 1

            if broken > 20:
                print("Potential bug")
            else:
                continue
    
    # Write results to file
    out_path = ROOT.joinpath(Path(f"data/json/{DSET}.json"))
    
    with out_path.open('w') as f:
        json.dump(results, f, ensure_ascii=False)
