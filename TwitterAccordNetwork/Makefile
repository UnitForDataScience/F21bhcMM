PYTHON3=${CONDA_PREFIX}/bin/python3

# Commands to create the Conda environments from the yml files
init: environment.yml
	conda env create -f environment.yml

init-sentiment: environment_sentiment.yml
	conda env create -f environment_sentiment.yml

# Commands to run each of the scripts necessary
pull:
	$(PYTHON3) src/pull_data.py

clean:
	$(PYTHON3) src/clean_data.py

train_word2vec:
	$(PYTHON3) src/train_word2vec.py

train_lda:
	$(PYTHON3) src/train_lda.py

# Note: this command requires the SentimentAnalysis environment to be active
# All others require the TwitterAccordNetwork environment to be active
analyze_sentiment:
	$(PYTHON3) src/sentiment_analysis.py

graph:
	$(PYTHON3) src/to_graph.py

analyze_modularity_groups:
	$(PYTHON3) src/modularity_analysis.py

# Commands to update and export current environments
update: environment.yml
	conda env update --name TwitterAccordNetwork --file environment.yml --prune

export:
	conda env export --name TwitterAccordNetwork --no-builds | grep -v "^prefix: " > environment.yml

update-sentiment: environment_sentiment.yml
	conda env update --name SentimentAnalysis --file environment_sentiment.yml --prune

export-sentiment:
	conda env export --name SentimentAnalysis --no-builds | grep -v "^prefix: " > environment_sentiment.yml
