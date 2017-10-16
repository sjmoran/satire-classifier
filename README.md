# A Satire Detection Model using Gaussian and Multinomial Naive Bayes

Current version: 1.0.0 Distributed under an [MIT License](https://opensource.org/licenses/MIT)

This code implements a satire detection machine learning model. The task is framed as a binary classification task where the goal is to predict the satire/non-satire labels for a set of documents.
A number of different feature types are explored:

1. Document embeddings using [word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
2. Unigram counts filtered by relevance to the class using the Chi2 score 
3. Punctuation counts
4. Capitalisation (lower, upper, mixed) counts
5. Sentiment polarity counts based on this [paper](http://www.aclweb.org/anthology/P15-2124)
6. [Intensifier](https://en.wikipedia.org/wiki/Intensifier) and [interjection](http://www.english-grammar-revolution.com/list-of-interjections.html) word counts

A [Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) model on discrete count-based features appears to perform very well for this task, and so more sophisticated models were not explored. The document embeddings are modelled using a [Gaussian Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) model. The probabilities output from the Multinomial and Naive Bayes models are combined by treating the class membership probabilities from both models as a new 4-dimensional feature space and learning another Gaussian Naive Bayes model on those features to predict the class labels.

The model achieves a mean 10-fold cross-validation FScore of 0.93, and a test set FScore of 0.74. This could be improved by carefully crafting more features (e.g. POS tagging features) and perhaps, to a lesser extent, using more powerful classifiers.

## Tackling the satire detection task

Satire detection is a challenging but relatively lightly explored NLP task. A satirical article is one which ridicules real-world individuals and organisations. It is a challenging machine learning task due to the frequent subtlety of the word-play that is characteristic of satire.

The satire detection task can be further broken down further along the axes of humour, irony and sarcasm detection, all of which are common components of [satire](https://en.wikipedia.org/wiki/Satire). Humour, irony and sarcasm detection are all non-trivial NLP tasks in and of themselves and have all attracted their own field of research. As a first attempt I would suggest tackling sarcasm detection problem by treating each sub-problem individually (i.e. crafting feature effective for each in isolation) before combining the output of three classifiers crafted for each sub-task using an ensembling technique.

Effective signals for automatic satire detection range from simple linguistic features (e.g. punctuation, presence of certain words), to more involved signals such as variation in sentiment, entity detection and resolution, part-of-speech tagging and grammar analysis. All of these methods of pure text analysis are limited in the sense that context and real-world knowledge is also invaluable for the task. It may be very challenging to integrate this type of information into a machine learning algorithm. For example the algorithm would need detailed understanding of the entitites in *"Bank Of England Governor Mervyn King is a
Queen, Says Fed Chairman Ben Bernanke"*, to ascertain that this sentence is absurd and likely from a satirical article. This signal cannot be gleaned from surface features alone.

The experiments in this repository explore how far simple linguistic features can get us on the task. The classifier is tested on the dataset of [Burfoot and Baldwin](http://www.aclweb.org/anthology/P09-2041). This is a small handpicked dataset and is likely to be of limited value for real-world applications. A real-world application of satire detection would require a much richer and larger training dataset. I would suggest gathering all the articles on reputable well-known satirical websites (e.g. The Onion) and using those as positive exemplars for evaluation. Furthermore, one could also crawl Facebook, Twitter, Reddit, Amazon reviews etc for posts explicitly labelled as satirical or one of its close relations (#sarcasm, #irony). Going further, creating a crowdsourcing platform that specifically encourages users to add high-quality labels to articles would be a particularly useful source of training data.

With a large enough dataset we could investigate using a convolutional neural network to automatically discover features for the task, removing the need for extensive feature hand-crafting. It would be interesting to compare this classifier to shallow models using hand-crafted features.

## Experimental Results

The model is tested on the dataset of [Burfoot and Baldwin](http://www.aclweb.org/anthology/P09-2041). The best model found via 10-fold cross-validation on the training datasets is applied to the test dataset. An ablation study is shown in the following table, alongside the test dataset results:

| Features      | Training (F1) | Testing (F1)|
| ------------- | ------------- |-------------|
| *Unigram (top 1000 chi2)*     | 0.86  |  0.66           | 
| + *Punctuation/capitalisation*   | 0.91  | 0.70           |
| + *Sentiment*              | 0.92  |  0.69           |
| + *Watch word list*              | 0.91  | 0.72             |
| + *Word2vec*              |  0.93    |  0.74           |

The unigram features reap most of the gain in FScore on this dataset. Punctuation/capitalisation features give another boost demonstrating their importance for the satire detection task (! tends to be discriminative of satirical articles). Sentiment features boost training performance by a small amount but hurt testing performance. The watch word (intensifier, interjection) feature hurts performance on the training dataset but benefits testing FScore. Finaly integrating the continuous word2vec features boosts training FScore (very lower dimensional word vectors - 10 dimensions - appear to be best here).

If we choose the features leading to the highest 10-fold cross-validation score (0.93) we would select all the features, which gives us a test set score of 0.74. The original paper by Burfoot and Baldwin obtain a best score of 0.78 on the same test set.

A limitation of this study is that the small gains in effectiveness after adding each feature would need to be cross-checked with a statistical significance test (e.g. t-test) to see if they are actually statistically significant.

The [DummyClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) which generates predictions uniformly at random performs the best out of all DummyClassifiers (uniform, stratified, most frequent etc). It scores a mean 0.10 FScore via 10-fold cross-validation on the training dataset using all the available feature sets combined. This sanity check shows that our informed classifier is performing better than random.

## Top-25 most discriminative features

The following 25 features were found to be most discriminative using a Chi2 test against the class label:

1. !
2. '
3. \*
4. \?
5. bush
6. exxon
7. fossil
8. hair
9. mat
10. mccain
11. mobil
12. mr
13. nato
14. obama
15. phelps
16. rutherford
17. slicked
18. tha
19. you
20. Sentiment_pos
21. Sentiment_neg
22. Token_upper
23. Token_lower
24. Token_mixed
25. Token_punctuation

## Prerequisites:

**Python 2.7** is required to run this code, amongst a list of other dependencies.
See the requirements.txt file for a list of dependencies. To install the requirements I recommend using python virtualenv:

1. pip install virtualenv
2. virtualenv venv
3. source ./venv/bin/activate
4. pip install -r requirements.txt 

You should now have all the dependencies needed to run the classifier straight away. Type deactivate to close the virtualenv after you have finished using the classifier.

## Usage

usage: train_classifier.py [-h] --training_dirpath TRAINING_DIRPATH
                           --testing_dirpath TESTING_DIRPATH --out_dirpath
                           OUT_DIRPATH --training_labels_filepath
                           TRAINING_LABELS_FILEPATH --testing_labels_filepath
                           TESTING_LABELS_FILEPATH --intensifier_data_filepath
                           INTENSIFIER_DATA_FILEPATH
                           
python train_classifier.py --testing_dirpath ./test --training_dirpath ./training --training_labels_filepath ./training-class --testing_labels_filepath ./test-class --out_dirpath ~/ --watch_words_data_filepath ./words.txt

## Copyright

Copyright (C) by Sean Moran

Please send any bug reports to sean.j.moran@gmail.com
