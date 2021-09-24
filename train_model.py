# Review model generator
# Author Misheel 
# Environment: Python 2.7, GraphLab
# Modified version of Machine Learning Specialization Course (University of Washington)

import argparse
import sys
import graphlab

def main(args): 

    #set graphlab config
    graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

    #parse CL arguments 
    parser = argparse.ArgumentParser(description='Review system')
    parser.add_argument('-f', '--file', default='udecideReview.csv', type=str, help='training data set')
    args = parser.parse_args()

    #load Data
    businesses = graphlab.SFrame(args.file) 
    businesses['word_count'] = graphlab.text_analytics.count_words(businesses['review'])

    # divide reviews into positive or negative
    businesses = businesses[businesses['rating'] != 3] # excluding reviews with value of 3.
    businesses['sentiment'] = businesses['rating'] >= 4 # positive reviews 1, negatives 0 

    # training
    train_data, test_data = businesses.random_split(.8, seed=0)

    # logistic classifier is for binary values such as negative, positive, hotdog, not hotdog etc... 

    sentiment_model = graphlab.logistic_classifier.create(train_data, 
                                                        target='sentiment',
                                                        features=['word_count'],
                                                        validation_set=test_data)

    # save model 
    sentiment_model.save('review_model');

    # evaluation of model 

    # roc_curve metric is graphical plot for binary classifier
    sentiment_model.evaluate(test_data, metric='roc_curve')
    # sentiment_model.show(view='Evaluation')
    

if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()