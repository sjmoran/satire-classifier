"""
Author:  Sean Moran
Contact: sean.j.moran@gmail.com
Date:    2017

MIT License: https://opensource.org/licenses/MIT
"""
import os
import logging
import argparse
import sys
import csv
import logging.config
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,StratifiedKFold, train_test_split
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer
from scipy import interp
from sklearn.svm import LinearSVC
import nltk
from nltk import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from gensim.models import Word2Vec
from nltk.corpus import stopwords as sw
from sklearn import linear_model
import random
import math
from collections import defaultdict
import pandas as pd
import time
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import traceback
import re
from sklearn.feature_extraction.text import CountVectorizer
import io
import errno
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest

'''
Download the ntlk files we need
'''
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
'''
Setup the logger to record sytsem output
'''
logging.config.fileConfig("./logging.conf")  
logger = logging.getLogger("classifier.dev")

class CrossValidationHelper:
        '''
        A helper class that performs grid search across multple models and metrics in parallel utilising as many cores as are available on the machine
        This code has been adapted from: http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
        '''
        def __init__(self, models, params):
            """Initialise the object with the models and parameters we wish to use in the grid search

            :param models: dictionary of scikit learn models
            :param params: dictionary of parameters for those models
            :returns: None
            :rtype: N/A

            """
            '''
            Initialise the object with the models and parameters we wish to use in the grid search
            '''
            if not set(models.keys()).issubset(set(params.keys())):
                        missing_params = list(set(models.keys()) - set(params.keys()))
                        raise ValueError("Some estimators are missing parameters: %s" % missing_params)
            self.models = models
            self.params = params
            self.keys = models.keys()
            self.grid_searches = {}

        def fit(self, X, y, scoring, n_folds=10, n_jobs=1, verbose=1, refit='FScore'):
            """ Performs n-fold cross-validation using the powerful GridSearchCV class

            :param X: training dataset
            :param y: training labels
            :param scoring: list of performance metrics to use e.g. FScore
            :param n_folds: number of folds (default 10) 
            :param n_jobs: number of jobs (default use all cores)
            :param verbose: true/false to print out verbose messages
            :param refit: refit best model on all the training data at the end (so it can be used for testing)
            :returns: dictionary mapping model to grid search results
            :rtype: dictionary

            """
            '''
            Performs grid search for multiple models and multiple evaluation metrics
            '''
            for key in self.keys:
                print("Running GridSearchCV for %s." % key)
                model = self.models[key]
                params = self.params[key]
                gs = GridSearchCV(model, params, cv=n_folds, n_jobs=n_jobs, verbose=verbose, scoring=scoring, refit=refit)
                gs.fit(X,y)
                self.grid_searches[key] = gs   

        def score_summary(self, scoring):
            """Retrieves the best scoring scikit learn models resulting from n-fold cross-validation

            :param scoring: the GridSearchCV scoring object containing the cross-validation results
            :returns: dictionary mapping model name to its highest scoring scikit learn model object
            :rtype: dictionary

            """
            '''
            This function returns the best parameterisation of a classifier for a given scoring metric 
            '''
            best_models_dict={}
            for k in self.keys:  # k is a classifier
                results=self.grid_searches[k].cv_results_ # results are all the results for classifier k
                for scorer in scoring:   # scorer represents one of the evaluation metrics for classifier k
                    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                    best_score = results['mean_test_%s' % scorer][best_index]
                    best_params = results['params'][best_index]
                    logger.info("Classifier: " + k)
                    logger.info("Evaluation Metric: " + scorer)
                    logger.info("Best score: " + '{0:.3f}'.format(best_score))
                    logger.info("Best parameter settings: ")
                    for key,value in best_params.iteritems():
                        logger.info(key + ":\t" + str(value))
                    '''
                    Save the best models for the given scoring metric specified by the refit parameter
                    '''
                best_models_dict[k]=self.grid_searches[k].best_estimator_

            return best_models_dict
    
class SatireClassifier:
       
    def __init__(self, training_dirpath, training_labels_filepath, testing_dirpath, testing_labels_filepath, out_dirpath, watch_words_data_filepath):
        """Initialises the classifier object with paths to the datasets and output directories

        :param training_dirpath: directory containing the training documents
        :param training_labels_filepath: file containing the training labels
        :param testing_dirpath: directory containing the testing documents
        :param testing_labels_filepath: file containing the testing labels
        :param out_dirpath: output directory for the results
        :param watch_words_data_filepath: file path to the watch words file
        :returns: None
        :rtype: N/A

        """
        self.training_dirpath=training_dirpath
        self.testing_dirpath=testing_dirpath
        self.out_dirpath=out_dirpath
        self.training_labels_filepath=training_labels_filepath
        self.testing_labels_filepath=testing_labels_filepath
        self.watch_words_data_filepath=watch_words_data_filepath
        
    def _cross_validate(self,training_features_df, n_folds, positive_weight, negative_weight, model='MultinomialNB'):
        """Cross validate using n_fold cross validation. Many models, metrics can be evaluated at once by defining a dictionary specifying their scikit learn names.

            :param training_text_df: pandas dataframe holding the training features 
            :param n_folds: number of cross validation folds 
            :param positive_weight: weight for the positive class
            :param negative_weight: weight for the negative class
            :returns: dictinary of best scoring models
            :rtype: dictionary
                   
        """
        logger.info("Performing grid search for the optimal model and parameters")

        '''
        I examine a broad collection of classifiers from scikit-learn. They are defined in a dictionary which is passed into the GridSearchCV function of scikit learn.
        '''
        if model in "GaussianNB":
                models = {
                        'DummyClassifier': DummyClassifier(),
                        'GaussianNB': GaussianNB(),
                }

                params = {
                        'DummyClassifier': { 'strategy': ["stratified", "most_frequent", "prior", "uniform"] },
                        'GaussianNB':  {'priors' : [None, [.1,.9],[.2, .8],[.3, .7],[.4, .6],[.5, .5],[.6, .4],[.7, .3],[.8, .2],[.9, .1]],
                        },}
        else:
                models = {
                        'DummyClassifier': DummyClassifier(),
                        'MultinomialNB': MultinomialNB(),
                }
                params = {
                        'DummyClassifier': { 'strategy': ["stratified", "most_frequent", "prior", "uniform"] },
                        'MultinomialNB': {'alpha': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 'class_prior' : [None, [.1,.9],[.2, .8],[.3, .7],[.4, .6],[.5, .5],[.6, .4],[.7, .3],[.8, .2],[.9, .1]]},
            }   
        
        '''
        I score based on F1 measure which is less sensitive to the class imbalance (very few satire, many non-satire documents).
        '''
        scoring = {'Precision': 'precision', 'Recall': 'recall', 'FScore': make_scorer(fbeta_score, beta=1.0)}  
        cross_val_helper = CrossValidationHelper(models, params)

        cross_val_helper.fit(training_features_df.loc[:,training_features_df.columns != 'Label'].values, training_features_df['Label'].values, scoring=scoring, n_jobs=-1, n_folds=n_folds)
        best_models_dict=cross_val_helper.score_summary(scoring)

        return best_models_dict

    def _get_word2vec_features(self, sentences, size=100, model=None):
        """Extracts word2vec features from a document. The words are averaged to form a single vector for a document. 

        :param sentences: list of documents
        :param size: dimension of the word vectors
        :param model: previously learnt word2vec model (optional)
        :returns: dictionary mapping a document name to its word vector
        :rtype: dictionary

        """
        embedding_df=pd.DataFrame()
        if not model:
                word2vec_sentences=[]
                for sentence in sentences:
                        average_vec=[]
                        words=sentence.split(" ")
                        word2vec_sentences.append(words)

                model = Word2Vec(word2vec_sentences, size=size, window=5, min_count=1, workers=4)

        columns=[]
        for i in range(0,size):
                columns.append(i)
        embedding_df=pd.DataFrame(columns=columns)

        row_id=0
        for sentence in sentences:
                average_vec=np.zeros(size)
                words=sentence.split(" ")
                count=0
                for word in words:
                        if word in model.wv.vocab:
                                average_vec+=model[word]
                                count+=1
                if count>0:
                        average_vec=average_vec/count
                embedding_df.loc[row_id]=average_vec
                row_id+=1
                
        return embedding_df, model


    def _add_watch_word_features_to_documents(self,text_df,doc_name_to_id_dict,watch_word_dict):
            """Adds the occurrence count of words in the user provided watch word list to the pandas dataframe 

            :param text_df: pandas data frame containing our text features
            :param doc_name_to_id_dict: dictionary mapping document name to row of the pandas dataframe 
            :param labels_dict: dictionary mapping a document name to label  
            :returns: pandas dataframe enriched with the watch word feature
            :rtype: pandas dataframe

            """
            for doc_name,row_id in doc_name_to_id_dict.iteritems():
                if doc_name in watch_word_dict:
                        watch_word_count=watch_word_dict[doc_name][0]
                        logger.debug("Word list count is: " + str(watch_word_count[0]) + " for document: " + doc_name)
                        text_df.ix[row_id,'Watch_word_count']=watch_word_count[0]
                else:
                        logger.debug("Could not find " + doc_name + " in the word_list_dict, even though it should really be there.")

            return text_df

    def _extract_watch_word_features_from_text(self, corpus_list, doc_name_to_id_dict):
        """Counts up the number of occurrences of watch words from the user provided watch word list in the documents
        :param corpus_list: list of documents
        :returns: dictionary mapping document name to the watch word count feature 
        :rtype: dictionary

        """
        '''
        Go through the documents and add up the number of occurrences of watch words
        '''
        doc_count=0
        watch_word_feature_dict=defaultdict(list)
        watch_words=io.open(self.watch_words_data_filepath, mode="r", encoding="ISO-8859-1").read()
        watch_word_list=watch_words.split("\n")

        for doc_name, row_id in doc_name_to_id_dict.iteritems():
            logger.debug("Extracting watch word features from: " + doc_name)
            doc=corpus_list[row_id]
            sentences=doc.split(".")
            watch_word_count=0
            for sentence in sentences:
                    words=sentence.split(" ")
                    for word in words:
                               if re.search('[a-zA-Z]',word):
                                       for watch_word in watch_word_list:
                                               if word.lower()==watch_word.lower():
                                                       watch_word_count+=1
            watch_word_feature_dict[doc_name].append([watch_word_count])
        return watch_word_feature_dict

    @staticmethod
    def tokenize(text):
        """Remove stopwords and tokenize the text

        :param text: text string representing a document from the corpus
        :returns: list of tokens
        :rtype: list

        """
        stemmer=PorterStemmer()
        stopwords  = set(sw.words('english'))

        text=text.replace('\n','')
        words=text.split(" ")
        filtered_text=[]
        for word in words:
                if word.lower() not in stopwords:
                        if len(word)>0:   
                                filtered_text.append(word)

        tokens = nltk.word_tokenize(' '.join(filtered_text))
        '''
        stemmed=[]
        for item in tokens:
              stemmed.append(stemmer.stem(item))
        '''
        return tokens
                  
    def _get_best_features(self, training_data, training_data_labels, testing_data, feature_names, number_top_features=10):
        """Interrogates the model to find out what features are more discriminative for the positive and negative classes
        Usually the computationally light chi2 test against the class labels.

        :param training_data: numpy ndarray representing the training dataset features
        :param training_data_labels: numpy ndarray representing the training dataset labels
        :param testing_data: numpy ndarray representing the testing dataset features  
        :param feature_names: list of feature names 
        :returns: list containing the number_top_features
        :rtype: list

        """
        logger.info("Ranking features using the chi2 test ..")

        ch2 = SelectKBest(chi2, k=number_top_features)
        training_data_filtered = ch2.fit_transform(training_data, training_data_labels)
        # keep selected feature names
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        feature_names.append('Label')

        return feature_names

    def _load_labels(self, labels_filepath):
        """Loads the labels from the labels file

        :param labels_filepath: path to the labels file
        :returns: dictionary mapping the document name to its class
        :rtype: dictionary

        """
        labels_dict={}
        labels_file = Path(labels_filepath)
        if labels_file.is_file():
            logger.info("Loading: " + labels_filepath)
            with open(labels_filepath, "r") as ins:
                for line in ins:
                    line=line.strip("\n")
                    line=line.split(" ")
                    file_name=line[0]
                    file_class=line[1]
                    labels_dict[file_name]=file_class
        else:
            raise IOError("Labels file: " + labels_filepath + " does not exist.")
        return labels_dict

    def _extract_token_features_from_text(self, corpus_list, doc_name_to_id_dict):
        """Extracts simple punctuation and capitalisation count-based features from documents

        :param corpus_list: list of documents
        :param doc_name_to_id_dict: dictionary mapping from document name to position in corpus_list
        :returns: dictionary mapping document name to a list of token features
        :rtype: dictionary

        """
        '''
        Go through the documents and extract simple punctuation and lexical 
        features (capitalisation, count of punctuation)
        '''
        doc_count=0
        token_feature_dict=defaultdict(list)
        for doc_name, row_id in doc_name_to_id_dict.iteritems():
            logger.debug("Extracting token features from: " + doc_name)
            doc=corpus_list[row_id]
            sentences=doc.split(".")
            upper_count=0
            lower_count=0
            mixed_count=0
            punctuation_count=0
            for sentence in sentences:
                    words=sentence.split(" ")
                    for word in words:
                       if word.isupper():
                               if re.search('[a-zA-Z]',word):
                                       upper_count+=1
                       if word.islower():
                               if re.search('[a-zA-Z]',word):
                                       lower_count+=1
                       if not word.islower() and not word.isupper():
                               if re.search('[a-zA-Z]',word):
                                       mixed_count+=1                     
                       if word in string.punctuation:
                               if len(word)>0:
                                       punctuation_count+=1
                               
            token_feature_dict[doc_name].append([upper_count,lower_count,mixed_count,punctuation_count])
        return token_feature_dict

    def _extract_sentiment_from_text(self, corpus_list, doc_name_to_id_dict):
        """Extracts several simple sentiment features from a document. I count the number of positive and negative sentiment words in a document, 
        the number, the count of the longest run of positives/negatives and the overall polarity of the document. These features are attempting 
        to identify incongruety, and therefore potentially sarcasm. See paper: Joshi et al. (2015), Harnessing Context Incongruity for Sarcasm Detection
        
        :param corpus_list: list of documents from corpus
        :param doc_name_to_id_dict: mapping from the document name to its position in corpus_list
        :returns: dictionary of sentiment features per document
        :rtype: dictionary

        """
        vader = SentimentIntensityAnalyzer()
        '''
        Go through the documents and rate their sentiment
        '''
        doc_count=0
        sentiment_feature_dict=defaultdict(list)
        for doc_name, row_id in doc_name_to_id_dict.iteritems():
            logger.debug("Extracting sentiment from: " + doc_name)
            doc=corpus_list[row_id]
            ''' 
            doc is one document from our corpus
            '''
            sentences=doc.split(".")
            pos_count=0
            neg_count=0
            prev_word_was_positive=False
            prev_word_was_negative=False
            pos_neg_count=0
            count=0
            longest_run_of_positives=0
            longest_run_of_negatives=0
            run_of_positives_count=0
            run_of_negatives_count=0
            score=vader.polarity_scores(' '.join(sentences))
            compound_polarity=score['compound']
            '''
            Rate the overall polarity of the document (1 positive, 0 negative)
            '''
            if compound_polarity>0:
                compound_polarity=1
            else:
                compound_polarity=0

            '''
            Rate each word in the corpus for sentiment and construct the word-based
            features
            '''
            for sentence in sentences:
                    words=sentence.split(" ")
                    for word in words:
                        score=vader.polarity_scores(word)
                        '''
                        If the negative sentiment of a word is greater than the positive sentiment
                        '''
                        if score['pos']>abs(score['neg']):
                                pos_count+=1
                                if prev_word_was_negative:
                                        pos_neg_count+=1
                                        prev_word_was_negative=False
                                        if run_of_negatives_count>longest_run_of_negatives:
                                                longest_run_of_negatives=run_of_negatives_count
                                                run_of_negatives_count=0
                                else:
                                        run_of_positives_count+=1
                                prev_word_was_positive=True

                        '''
                        If the positive sentiment of a word is greater than the negative sentiment
                        '''
                        if score['pos']<abs(score['neg']):
                                neg_count+=1
                                if prev_word_was_positive:
                                        prev_word_was_positive=False
                                        pos_neg_count+=1
                                        if run_of_positives_count>longest_run_of_positives:
                                                longest_run_of_positives=run_of_positives_count
                                                run_of_negatives_count=0
                                else:
                                        run_of_negatives_count+=1
                                prev_word_was_negative=True
                        count+=1

            sentiment_feature_dict[doc_name].append([pos_count,neg_count,pos_neg_count,longest_run_of_negatives,longest_run_of_positives,compound_polarity])
            
        return sentiment_feature_dict
    
    def _load_text(self, data_dirpath, vectorizer=None):
        """ Parses and tokenises the input document files

        :param data_dirpath: the directory containing the documents
        :param vectorizer: scikit vectorizer to extract corpus counts (optional)
        :returns: vectorizer, pandas dataframe containing features, list of documents (corpus_list), mapping of document names to positions in the list of documents (corpus_list)
        :rtype: vectorizer, pandas dataframe, list, dictionary

        """
        corpus_list=[]
        document_name_to_id_dict={}
        count=0
        file_list=sorted(os.listdir(data_dirpath)) # read the files in sorted order
        for filename in file_list:
                data_filepath=data_dirpath+"/"+filename
                logger.debug("Loading: " + data_filepath)
                '''
                load in the document be mindful of the encoding
                '''
                text=io.open(data_filepath, mode="r", encoding="ISO-8859-1").read()
                tokens=SatireClassifier.tokenize(text)
                '''
                corpus_list is a list of the documents pre-processed for stopwords etc
                '''
                corpus_list.append(' '.join(tokens))
                '''
                dictionary that maps a filename to its position in corpus_list 
                '''
                document_name_to_id_dict[filename]=count
                count+=1

        '''
        Extract count features from the text
        '''
        if not vectorizer:
                '''
                We have not passed in a vectorizer, so create one. Else transform the dataset using the provided vectorizer e.g. so the training and testing datasets share the same words.
                '''
                vectorizer = CountVectorizer(ngram_range=(1,1),token_pattern=r"(?u)\b\w\w+\b|\*|!|\?|\"|\'", encoding="ISO-8859-1",strip_accents='unicode')
                '''
                vectorizer = TfidfVectorizer(ngram_range=(1,1),token_pattern=r"(?u)\b\w\w+\b|\*|!|\?|\"|\'", encoding="ISO-8859-1",strip_accents='unicode')
                TfidfVectorizer(sublinear_tf=True, max_df=0.75, stop_words='english')
        '''
                corpus_counts = vectorizer.fit_transform(corpus_list)
        else:
                corpus_counts = vectorizer.transform(corpus_list)
        '''
        Store the features and column names in a pandas dataframe for ease of manipulation. The words in the corpus are the column headings.
        '''
        corpus_counts_df = pd.DataFrame(corpus_counts.toarray(), columns=vectorizer.get_feature_names())
        return vectorizer,corpus_counts_df, corpus_list, document_name_to_id_dict

    def _add_labels_to_documents(self,text_df,doc_name_to_id_dict,labels_dict):
        """Adds satire/non-satire labels to the correct documents in the pandas dataframe

           :param text_df: pandas dataframe containing the text features for our documents
           :param doc_name_to_id_dict: dictionary mapping document name 
           :param labels_dict: 
           :returns: pandas dataframe enriched with class labels 
           :rtype: pandas dataframe 

        """
        logger.info("Adding labels to documents ...")
        for doc_name,row_id in doc_name_to_id_dict.iteritems():
                if doc_name in labels_dict:
                        label=labels_dict[doc_name]
                        logger.debug("Label is: " + label + " for document: " + doc_name)
                        if "true" in label:
                                text_df.ix[row_id,'Label']=0
                        else:
                                text_df.ix[row_id,'Label']=1
                else:
                        logger.debug("Could not find " + doc_name + " in the labels_dict, even though it should really be there.")
        return text_df

    def _add_sentiment_to_documents(self,text_df,doc_name_to_id_dict,sentiment_dict):
        """Adds the sentiment features to the pandas features dataframe

           :param text_df: pandas dataframe containing our features
           :param doc_name_to_id_dict: dictionary mapping document name to row in the dataframe
           :param sentiment_dict: dictionary containing the sentiment features for each document
           :returns: enriched dataframe containing the additional featureset
           :rtype: pandas dataframe

        """
        for doc_name,row_id in doc_name_to_id_dict.iteritems():
                if doc_name in sentiment_dict:
                        sentiment=sentiment_dict[doc_name][0]
                        logger.debug("Positive sentiment is: " + str(sentiment[0]) + " for document: " + doc_name)
                        logger.debug("Negative sentiment is: " + str(sentiment[1]) + " for document: " + doc_name)       
                        text_df.ix[row_id,'Sentiment_pos']=sentiment[0]
                        text_df.ix[row_id,'Sentiment_neg']=sentiment[1]
                        text_df.ix[row_id,'Sentiment_change']=sentiment[2]
                        text_df.ix[row_id,'Sentiment_neg_run']=sentiment[3]
                        text_df.ix[row_id,'Sentiment_pos_run']=sentiment[4]
                        text_df.ix[row_id,'Sentiment_compound']=sentiment[5]
                else:
                        logger.debug("Could not find " + doc_name + " in the labels_dict, even though it should really be there.")

        return text_df

    def _add_token_features_to_documents(self,text_df,doc_name_to_id_dict,tokens_dict):
        """Adds the token features to the pandas features dataframe

           :param text_df: pandas dataframe containing our features
           :param doc_name_to_id_dict: dictionary mapping document name to row in the dataframe
           :param tokens_dict: dictionary containing the token features for each document
           :returns: enriched dataframe containing the additional featureset
           :rtype: pandas dataframe

        """
        for doc_name,row_id in doc_name_to_id_dict.iteritems():
                if doc_name in tokens_dict:
                        token_features=tokens_dict[doc_name][0]     
                        text_df.ix[row_id,'Token_upper']=token_features[0]
                        text_df.ix[row_id,'Token_lower']=token_features[1]
                        text_df.ix[row_id,'Token_mixed']=token_features[2]
                        text_df.ix[row_id,'Token_punctuation']=token_features[3]
                else:
                        logger.debug("Could not find " + doc_name + " in the tokens_dict, even though it should really be there.")
        return text_df
      
    def _normalise_sparse_features(self,text_features_df,scaler=None):
        """Normalises sparse features so that their maximum is 1.0 while retaining sparsity. 

        :param text_features_df: pandas dataframe containing the text features
        :param scaler: scikit scaler to perform the normalisation (optional)
        :returns: normalised text features in pandas dataframe, scikit scaler
        :rtype: pandas dataframe, scikit scaler

        """
        text_features_without_labels=text_features_df.loc[:,text_features_df.columns != 'Label'].values
        if not scaler:
                scaler = preprocessing.MaxAbsScaler().fit(text_features_without_labels)  
        text_features_without_labels=scaler.transform(text_features_without_labels)
        text_features_df.loc[:,text_features_df.columns!='Label']=text_features_without_labels
        return text_features_df, scaler

    def _load_discrete_data(self):
        """Reads all the files in the data directory into a python list. I assume they contain text. This list is passed in
        to the scikit-learn CountVectorizer to extract tf-idf count-based features. CountVectorizer also stops, stems the 
        text. All the features are discrete counts.

        :returns: pandas dataframe for training features, training labels dictionary, training sentences list, pandas dataframe for testing features, testing labels dictionary, testing sentences list
        :rtype: pandas dataframe, dictionary, list, pandas dataframe, dictionary, list

        """
        '''
        Read the data 
        '''
        training_labels_dict=self._load_labels(self.training_labels_filepath)
        testing_labels_dict=self._load_labels(self.testing_labels_filepath)
        
        vectorizer,training_text_df,training_sentences,training_doc_name_to_id_dict=self._load_text(self.training_dirpath)
        training_sentiment_feature_dict=self._extract_sentiment_from_text(training_sentences, training_doc_name_to_id_dict)
        training_token_feature_dict=self._extract_token_features_from_text(training_sentences, training_doc_name_to_id_dict)
        training_watch_word_feature_dict=self._extract_watch_word_features_from_text(training_sentences, training_doc_name_to_id_dict)
    
        _,testing_text_df,testing_sentences,testing_doc_name_to_id_dict=self._load_text(self.testing_dirpath, vectorizer)
        testing_sentiment_feature_dict=self._extract_sentiment_from_text(testing_sentences, testing_doc_name_to_id_dict)
        testing_token_feature_dict=self._extract_token_features_from_text(testing_sentences, testing_doc_name_to_id_dict)
        testing_watch_word_feature_dict=self._extract_watch_word_features_from_text(testing_sentences, testing_doc_name_to_id_dict)
   
        logger.info("Size of training dataset: " + str(training_text_df.shape[0])+"x"+str(training_text_df.shape[1]))
        logger.info("Size of testing dataset: " + str(testing_text_df.shape[0]) +"x"+ str(testing_text_df.shape[1]))
        
        '''
        Merge the training labels into the training and testing dataset pandas dataframes
        '''
        training_text_df=self._add_labels_to_documents(training_text_df,training_doc_name_to_id_dict, training_labels_dict)
        testing_text_df=self._add_labels_to_documents(testing_text_df,testing_doc_name_to_id_dict, testing_labels_dict)

        logger.info("Size of training dataset: " + str(training_text_df.shape[0])+"x"+str(training_text_df.shape[1]))
        logger.info("Size of testing dataset: " + str(testing_text_df.shape[0]) +"x"+ str(testing_text_df.shape[1]))
        
        feature_names=self._get_best_features(training_text_df.loc[:,training_text_df.columns != 'Label'].values, training_text_df['Label'].values, testing_text_df.loc[:,testing_text_df.columns != 'Label'].values, training_text_df.loc[:,training_text_df.columns != 'Label'].columns.values, number_top_features=1000)
        '''
        Filter the training and testing datasets according to the best features found in get_best_features
        '''
        training_text_df=training_text_df[feature_names]
        testing_text_df=testing_text_df[feature_names]
        
        logger.info("Size of training dataset: " + str(training_text_df.shape[0])+"x"+str(training_text_df.shape[1]))
        logger.info("Size of testing dataset: " + str(testing_text_df.shape[0]) +"x"+ str(testing_text_df.shape[1]))

        '''
        Try out the experimental sentiment, token and intensifier/interjection features (they seem to help the task)
        '''        
        training_text_df=self._add_sentiment_to_documents(training_text_df,training_doc_name_to_id_dict, training_sentiment_feature_dict)
        testing_text_df=self._add_sentiment_to_documents(testing_text_df,testing_doc_name_to_id_dict, testing_sentiment_feature_dict)
        
        training_text_df=self._add_token_features_to_documents(training_text_df,training_doc_name_to_id_dict, training_token_feature_dict)
        testing_text_df=self._add_token_features_to_documents(testing_text_df,testing_doc_name_to_id_dict, testing_token_feature_dict)
          
        testing_text_df=self._add_watch_word_features_to_documents(testing_text_df,testing_doc_name_to_id_dict, testing_watch_word_feature_dict)
        training_text_df=self._add_watch_word_features_to_documents(training_text_df,training_doc_name_to_id_dict, training_watch_word_feature_dict)
     
        '''
        Normalise the count based features using MaxAbsScaler which is recommended for sparse feature sets
        training_text_df,scaler=self._normalise_sparse_features(training_text_df)
        testing_text_df,_=self._normalise_sparse_features(testing_text_df,scaler)
        '''
        return training_text_df,training_doc_name_to_id_dict,training_labels_dict,training_sentences,testing_text_df,testing_doc_name_to_id_dict,testing_labels_dict,testing_sentences

    def _test(self,testing_features_df,best_models_dict):
        """Computes the f1 score on a test dataset using the best model found during cross-validation

        :param testing_text_df: pandas dataframe containing the testing dataset features and labels
        :param best_models_dict: dictionary holding the best models found during cross validation 
        :returns: prints the f1-score to the screen/log file
        :rtype: None

        """
        best_model=best_models_dict['GaussianNB']
        pred=best_model.predict(testing_features_df.loc[:,testing_features_df.columns != 'Label'].values)
        score=metrics.f1_score(testing_features_df['Label'].values,pred)
        logger.info("F1-score on the testing dataset: " + str('{0:.2f}'.format(score)))

    def _load_continuous_data(self,training_sentences,training_doc_name_to_id_dict, training_labels_dict, testing_sentences, testing_doc_name_to_id_dict, testing_labels_dict):
        """Extracts continuous features from our documents, in this case word2vec features

        :param training_sentences: list of training sentences
        :param training_doc_name_to_id_dict: dictionary mapping document name to position in the sentences list
        :param training_labels_dict: dictionary mapping document name to its corresponding label
        :param testing_sentences: list of testing sentences
        :param testing_doc_name_to_id_dict: dictionary mapping document name to position in the sentences list
        :param testing_labels_dict: dictionary mapping document name to its corresponding label
        :returns: pandas dataframes representing the continuous features extracted from the training and testing datasets 
        :rtype: pandas dataframe, pandas dataframe

        """
        training_embedding_df,model=self._get_word2vec_features(training_sentences,size=10)
        testing_embedding_df,_=self._get_word2vec_features(testing_sentences,size=10,model=model)
        training_embedding_df=self._add_labels_to_documents(training_embedding_df,training_doc_name_to_id_dict, training_labels_dict)
        testing_embedding_df=self._add_labels_to_documents(testing_embedding_df,testing_doc_name_to_id_dict, testing_labels_dict)

        return training_embedding_df, testing_embedding_df    
        
    @staticmethod
    def get_class_weights(positive_count, negative_count):
        """Computes the class weights based on the number of positive and negative exemplars in the training dataset
        Can be useful for classifiers such as the SVM so as to balance the classes when learning the hyperplanes.

        :param positive_count: the number of positive exemplars in the training dataset
        :param negative_count: the number of negatives
        :returns: positive weight, negative weight
        :rtype: float, float

        """
        logger.info("Number of positive exemplars: " + str(positive_count))
        logger.info("Number of negative exemplars: " + str(negative_count))

        if positive_count>=negative_count:
            negative_weight=int(float(positive_count)/float(negative_count))
            positive_weight=1
        else:
            positive_weight=int(float(negative_count)/float(positive_count))
            negative_weight=1

        logger.info("Positive class weight: " + str(int(positive_weight)))
        logger.info("Negative class weight: " + str(int(negative_weight)))

        return positive_weight, negative_weight


    def train(self, experiment_name, n_folds=10):
        """ Train and test models on the satire detection task

        :param experiment_name: the name of the experiment for identification
        :param n_folds: number of cross-validation folds (default 10)
        :returns: N/A
        :rtype: N/A

        """
        try:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")   # timestamp for the directory name
                    out_dirpath = self.out_dirpath+"/"+experiment_name+"_"+timestamp
                    '''
                    Make the output directory if it doesnt exist
                    '''
                    if not os.path.exists(self.out_dirpath):
                        os.makedirs(self.out_dirpath)

                    '''
                    Extract text features and load the training and testing datasets into pandas dataframes
                    '''
                    training_text_df,training_doc_name_to_id_dict,training_labels_dict,training_sentences,testing_text_df,testing_doc_name_to_id_dict,testing_labels_dict,testing_sentences=self._load_discrete_data()

                    training_embedding_df,testing_embedding_df=self._load_continuous_data(training_sentences,training_doc_name_to_id_dict, training_labels_dict, testing_sentences, testing_doc_name_to_id_dict, testing_labels_dict)

                    positive_count=training_text_df[training_text_df['Label']==1].shape[0]
                    negative_count=training_text_df[training_text_df['Label']==0].shape[0]

                    positive_weight, negative_weight = SatireClassifier.get_class_weights(positive_count, negative_count)
                    '''
                    My goal now is to fuse the continuous and discrete features for the classification task. To so so I take a simple approach using Gaussian and Multinomial
                    Naive Bayes
                    '''
                    '''
                    I first traing a GaussianNB model on the continuous word2vec features. http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
                    '''    
                    '''
                    Use 10-fold cross-validation to pick the most performant model for the task
                    '''                        
                    best_models_dict=self._cross_validate(training_embedding_df, n_folds, positive_weight, negative_weight, 'GaussianNB')
                    training_continuous_data_probs=best_models_dict['GaussianNB'].predict_proba(training_embedding_df.loc[:,training_embedding_df.columns!='Label'])
                    testing_continuous_data_probs=best_models_dict['GaussianNB'].predict_proba(testing_embedding_df.loc[:,testing_embedding_df.columns!='Label'])

                    '''
                    Now I train a MultinomialNB model on the discrete text features
                    '''
                    best_models_dict=self._cross_validate(training_text_df, n_folds, positive_weight, negative_weight, 'MultinomialNB')
                    training_discrete_data_probs=best_models_dict['MultinomialNB'].predict_proba(training_text_df.loc[:,training_text_df.columns!='Label'])
                    testing_discrete_data_probs=best_models_dict['MultinomialNB'].predict_proba(testing_text_df.loc[:,testing_text_df.columns!='Label'])

                    '''
                    Use the trainined Gaussian and Multinomial NB models to annotate each training document with their probabilities of being in the positive and
                    negative classes.
                    '''
                    training_probs_features=np.concatenate([training_continuous_data_probs,training_discrete_data_probs],axis=1)
                    training_probs_features_df=pd.DataFrame(training_probs_features,columns=["GaussianNB_0","GaussianNB_1","MultinomialNB_0","MultinomialNB_1"])
                    testing_probs_features=np.concatenate([testing_continuous_data_probs,testing_discrete_data_probs],axis=1)
                    testing_probs_features_df=pd.DataFrame(testing_probs_features,columns=["GaussianNB_0","GaussianNB_1","MultinomialNB_0","MultinomialNB_1"])

                    '''
                    Concatenate the probabilities to create a 4-dimensional feature vector per document. I now train a new Gaussian NB model to combine these
                    probabilities to get an overall estimate of the class occupancy (this is a simple form of ensembling).
                    '''
                    training_probs_features_df=self._add_labels_to_documents(training_probs_features_df,training_doc_name_to_id_dict, training_labels_dict)
                    testing_probs_features_df=self._add_labels_to_documents(testing_probs_features_df,testing_doc_name_to_id_dict, testing_labels_dict)

                    best_models_dict=self._cross_validate(training_probs_features_df, n_folds, positive_weight, negative_weight, 'GaussianNB')
                    '''
                    Run the best model once on the testing dataset reporting the result
                    '''
                    self._test(testing_probs_features_df,best_models_dict)
                    
        except Exception, err:

            print Exception, err                                  
            print traceback.print_stack()                         
            logger.error(traceback.print_stack())                 
            exc_type, exc_obj, exc_tb = sys.exc_info()            
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print exc_type, fname, exc_tb.tb_lineno


                 
if __name__ == '__main__':
    """
        Main method
        Parameters
        ----------
        None
    """
    parser = argparse.ArgumentParser(
        description="Train a binary classifier on the satire/no satire task")

    parser.add_argument(
            "--training_dirpath", required=True, type=str, help="the training data directory")
    parser.add_argument(
            "--testing_dirpath", required=True, type=str, help="the testing data directory")
    parser.add_argument(
            "--out_dirpath", required=True, type=str, help="the directory to save the classifier output")
    parser.add_argument(
            "--training_labels_filepath", required=True, type=str, help="the training labels file")
    parser.add_argument(
            "--testing_labels_filepath", required=True, type=str, help="the testing labels")
    parser.add_argument(
            "--watch_words_data_filepath", required=True, type=str, help="file containing the watch words list")

    args = parser.parse_args()
    training_dirpath = args.training_dirpath
    testing_dirpath = args.testing_dirpath
    out_dirpath = args.out_dirpath
    testing_labels_filepath=args.testing_labels_filepath
    training_labels_filepath=args.training_labels_filepath
    watch_words_data_filepath=args.watch_words_data_filepath
    
    random.seed(1243)
    
    '''
    Train a satire classifier on labelled data
    '''
    satire_classifier = SatireClassifier(training_dirpath, training_labels_filepath, testing_dirpath, testing_labels_filepath, out_dirpath, watch_words_data_filepath)
    satire_classifier.train("factmata")
    
    
