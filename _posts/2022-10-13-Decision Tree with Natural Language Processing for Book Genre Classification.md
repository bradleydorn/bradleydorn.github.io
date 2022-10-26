**Decision Tree**

Predicts a variety of classes by subdividing data using cutoff thresholds until all observations in a subset fall into a single class
Decisions are binary, but multiple decisions are chained together to create many possible outcomes.

Easy to interpret, but high chance of overfitting.

- Related topic: Ensembles: create multiple models and aggregate results
- Bootstrapping: train on multiple subsamples, aggregate result to reduct overfitting
- Random forrest: train using a subset of predictors
- Boosting: fit a decision tree, then fit another tree to residuals of previous tree with more data 


**Natural Language Processing**

In addition to using a random forrest for classification, this script uses natural language processing techniques to manage data used in the classification process. The following techniques are used in this script:

- [Term frequency inverse document frequency (TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) 
- [Sentiment analysis](https://realpython.com/python-nltk-sentiment-analysis/#using-nltks-pre-trained-sentiment-analyzer)
- [Word stemming (Porter's)](https://www.nltk.org/api/nltk.stem.porter.html) 
- [Stopword removal](https://pythonspot.com/nltk-stop-words/) 


```python
#Libraries for data handling and decision tree
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
```


```python
#Libraries for natural language processing
from nltk import word_tokenize
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
ps = PorterStemmer()
sia = SentimentIntensityAnalyzer()
tfidf = TfidfVectorizer()
```


```python
# Configurable parameters for decision tree and testing
n_esimators = 250 # number of trees in the forrest
max_depth = 200 # maximum number of splits
min_samples_split = 4 # minimum number of samples required for split nodes
min_samples_leaf = 1 # mimum number of samples required to be a leaf - each split must have this many
max_features = None # maxmimum number of features to consider when looking for the best split - each leaf must have this many 
bootstrap = True # default to true, but can be set to false
max_samples = None # default is None, if int, draw int number of samples in bootstrapping, if float, then (from 0 to 1) draw a percentage of samples
criterion = 'log_loss' #options are gini, entropy, and log_loss
random_state = 40
test_size = 0.1
use_synopsis = True
use_ratings = True
use_sentiment = True
use_title = True

# Output parameters for readability
# pd.set_option('display.height', 10)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 20)

# Configurable parameters for natural language processing 
keep_counts_min = 8 # Keep word stems as features if they occur this many times or more in the dataset
keep_counts_max = 500 # Keep word stems as features if they occur this many times or fewer in the dataset
```

**Load Data**

The data used in this script features synopses and titles of books with a known genre. The goal of this script is to create a classifier based on contents of the synopses (and possibly titles) to classify the genre of unknown (or hypothetically unknown by being sampled out through test/train split) books. 

The following section considers the possible use of other variables also included in the dataset. 


```python
#Example code uses the TagMyBook Kaggle dataset: https://www.kaggle.com/datasets/athu1105/tagmybook
df = pd.read_csv('data/tagmybook.csv')
```


```python
#Data configurable parameters - NLP parameters are added later
# unused variables: 'title', 'name', 'synopsis',, 'num_ratings' 'num_followers' ' num_reviews'
x_vars = []
y_var = 'genre'

if use_ratings:
	x_vars.extend(['rating','num_ratings','num_followers','num_reviews'])
```

**Natural Language Processing**

This section of the script creates variables for later use in the classification model.


```python
# Clean commas and other irrelevant information from numeric columns
for x in x_vars:
	df[x] = [float(str(y).replace(",","").replace('k','').replace(' followers','')) for y in df[x]]

# If you want to use the title, add title to the synopsis
if use_title:
	df['synopsis'] = [df['title'][i] + ' ' + df['synopsis'][i] for i in range(len(df['synopsis']))]

# Natural Language Processing (precedes decision trees)-------------------
# Overview: tokenize and stem words, select tokens to use, calculate tf-idf 

# Remove stopwords (words that are frequently too common to be useful)
stopwords = stopwords.words("english")

# Tokenize words, remove stopwords, stem, and convert to lower case all words
tokenized_synopses = [word_tokenize(df['synopsis'][i]) for i in range(0,len(df['synopsis']))]
for i in range(0,len(tokenized_synopses)):
	tokenized_synopses[i] = [ps.stem(x.lower()) for x in tokenized_synopses[i] if x not in stopwords] #Stem

# Add tokens back to dataframe as token arrays and as space-separated tokens
df['synopsis_words_array'] = tokenized_synopses
df['synopsis_words'] = [" ".join(tokenized_synopses[i]) for i in range(len(tokenized_synopses))]

# Calculate sentiment scores
sentiments = [sia.polarity_scores(df['synopsis'][i]) for i in range(0,len(df['synopsis']))]
df['synopsis_positive'] = [sentiments[i]['pos'] for i in range(0,len(df['synopsis']))]
df['synopsis_negative'] = [sentiments[i]['neg'] for i in range(0,len(df['synopsis']))]
df['synopsis_neutral'] = [sentiments[i]['neu'] for i in range(0,len(df['synopsis']))]
df['synopsis_compound'] = [sentiments[i]['compound'] for i in range(0,len(df['synopsis']))]

# Add sentiment scores to list of x variables
if use_sentiment:
	x_vars.extend(['synopsis_positive','synopsis_negative','synopsis_neutral','synopsis_compound'])

# Remove tokens that are too common or not common enough
# Determine unique tokens and calculate frequencies
if use_synopsis:
	all_tokens = np.concatenate(tokenized_synopses)
	word_frequencies = FreqDist(all_tokens)
	counts = list(word_frequencies.values())
	stems = list(word_frequencies.keys())

	# Remove tokens that occur too frequently or not frequently enough
	re_index_rate = 70
	kept_stems = []
	kept_counts = []
	for x in range(0,len(counts)):
		if counts[x] >= keep_counts_min and counts[x] <= keep_counts_max:
			kept_stems.append(stems[x])
			kept_counts.append(counts[x])

	# Calculate tfidf and add it back to the dataframe
	# Calculate tfidf
	tfidf_tokens = tfidf.fit_transform(df['synopsis_words'])
	tf_idf_names = tfidf.get_feature_names_out()
	tfidf_tokens_array = tfidf_tokens.toarray()

	#Only keep tf-idf scores for tokens that meet thresholds
	j = 0
	for x in range(0,len(tf_idf_names)):
		if tf_idf_names[x] in kept_stems:
			df[tf_idf_names[x]] = [tfidf_tokens_array[i][x] for i in range(len(df['synopsis_words']))]
		# x_vars.append(x_stem)
		# df[x_stem] = [df['synopsis'][i].count(x_stem) for i in range(0,len(df['synopsis']))]
		if j == re_index_rate: #Periodically re-index the data frame because we may be adding many columns
			j = 0
			df = df.copy()
		j+=1
	df = df.copy()

	#Remove x-variables if they are not in the tfidf set
	kept_stems = [x for x in kept_stems if x in tf_idf_names]

	x_vars.extend(kept_stems) # add kept tokens to list of predictor variables

print("Number of predictors: " + str(len(x_vars)))
```

    Number of predictors: 3189
    

**Model Fitting and Evaluation**

Now we split the data into training and test data, fit the model and evaluate accuracies overall and by genre. 


```python
# Decision Tree using linguistic features calculated in previous section-------------------
training_data, testing_data = train_test_split(df,test_size = test_size, random_state = random_state) # hold out a subset of the data for testing
```


```python
# Fit the model
random_forrest = RandomForestClassifier(n_estimators=n_esimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = max_features, bootstrap = bootstrap, max_samples = max_samples)
random_forrest.fit(training_data[x_vars],training_data[y_var])
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(criterion=&#x27;log_loss&#x27;, max_depth=200, max_features=None,
                       min_samples_split=4, n_estimators=250)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(criterion=&#x27;log_loss&#x27;, max_depth=200, max_features=None,
                       min_samples_split=4, n_estimators=250)</pre></div></div></div></div></div>




```python
# Calculate overall model accuracy on test data
y_pred=random_forrest.predict(testing_data[x_vars]) 
print("Overall Accuracy:",metrics.accuracy_score(testing_data[y_var], y_pred)) 

# Calculate accuracies for each class in the test data
accuracies = []
classes = pd.unique(training_data[y_var]) #
for x in classes:
	itesting = testing_data[testing_data[y_var] == x]
	iy_pred = random_forrest.predict(itesting[x_vars])
	accuracies.append([x,": ",metrics.accuracy_score(itesting[y_var], iy_pred)])
print("Accuracy by class:")
print(accuracies)
```

    Overall Accuracy: 0.6753246753246753
    Accuracy by class:
    [['psychology', ': ', 0.4], ['fantasy', ': ', 0.75], ['thriller', ': ', 0.9534883720930233], ['travel', ': ', 0.7692307692307693], ['science', ': ', 0.6428571428571429], ['sports', ': ', 0.5], ['horror', ': ', 0.2], ['romance', ': ', 0.3076923076923077], ['history', ': ', 0.7777777777777778], ['science_fiction', ': ', 0.16666666666666666]]
    

It seems that classification accuracy varies widely by genre. For example, 95% of thriller books are classified correctly, but only 17% of horror books. Next steps to improve the model would include investigating reasons why certain genres seem to perform worse than others. Possible explanations include unbalanced data or lack of distinct keywords for certain genres. Through investigating reasons that certain genres are not classified accurately by this model I would hope to discover additional and/or better variables to include (for example: removing certain tokens or creating variables that represent phrases that are common to genres in addition to just specific words). 
