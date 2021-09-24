# Make prediction using pre-trained model 

import argparse
import sys
import graphlab

def main(args):

    #set graphlab config
    graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

    #parse CL arguments 
    parser = argparse.ArgumentParser(description='Review system model loader')
    parser.add_argument('-m', '--model', default='review_model', type=str, help='trained model')
    parser.add_argument('-f', '--file', default='udecideReview.csv', type=str, help='training data set')
    args = parser.parse_args()

    # to load model and data
    sentiment_model = graphlab.load_model(args.model)
    reviews = graphlab.SFrame(args.file) 

    print 'Model loaded successfully'

    # predict 

    reviews['predicted_sentiment'] = sentiment_model.predict(reviews, output_type='probability')
    print reviews[0]


if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()