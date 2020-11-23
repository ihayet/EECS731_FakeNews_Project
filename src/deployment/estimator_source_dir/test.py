import numpy as np
import pandas as pd

import os
import argparse
import json

from sklearn.externals import joblib

# ----------------------------------------------------------------------------------------------------------------------------
# Functions for training approaches - start

# Kaggle PAC classifier
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stopword = stopwords.words('english')
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

def kaggleFakeNewsClassifier(data):
    data = data.iloc[0:20]
    data = data.dropna()

    ps = PorterStemmer()

    max_features_val = 100
    tfidfVectorizer = TfidfVectorizer(max_features = max_features_val, ngram_range = (1, 3))

    cleanedTitle = []
    cleanedText = []
    for i in range(0, len(data)):
        title = data['title'].iloc[i]
        text = data['text'].iloc[i]  
    
        title = re.sub('[^a-zA-z]', ' ', title)
        text = re.sub('[^a-zA-z]', ' ', text)

        title = title.lower()
        text = text.lower()

        title = title.split()
        text = text.split()

        title = [ps.stem(word) for word in title if not word in stopword]
        text = [ps.stem(word) for word in text if not word in stopword]

        title = ' '.join(title)
        text = ' '.join(text)

        cleanedTitle.append(title)
        cleanedText.append(text)

    X_train = tfidfVectorizer.fit_transform(cleanedTitle).toarray()
    y_train = data['label']

    paClassifier = PassiveAggressiveClassifier()
    paClassifier.fit(X_train, y_train)

    pipeline = {
        'tfidf': tfidfVectorizer,
        'model': paClassifier
    }

    return pipeline


from FeatureExtractor import Extractor

def extractFeatures(data):
    extractor = Extractor(data)
    data = extractor.extractFeatures()

# Kaggle/Popularity classifier
def kagglePopularityClassifier(data):
    return


# Kaggle/Popularity regressor
def kagglePopularityRegressor(data):
    return


# Kaggle/Popularity clustering
from sklearn.cluster import KMeans
from kneed import KneeLocator

def elbowDetector(dataFeatures, max):
    wcss =[]
    for j in range (1, max):
        kmeans = KMeans(n_clusters = j, init = 'k-means++', max_iter =100, n_init = 10, random_state = 0)
        kmeans.fit(dataFeatures)
        wcss.append(kmeans.inertia_)
    
    kn = KneeLocator(range(1, max), wcss, curve='convex', direction='decreasing')
    elbow_k=kn.knee

    return elbow_k

def kagglePopularityRecommender(data, dataFeatures):
    kmeans = KMeans(n_clusters=elbowDetector(dataFeatures, 20), init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0)
    index_clusters = kmeans.fit_predict(dataFeatures)

    data['clusters'] = index_clusters
    data = data[data['label']==0]

    print(data.columns)
    data = data[['id', 'author', 'title', 'clusters']]

    pipeline = {
        'data_ref': data,
        'model': kmeans
    }

    return pipeline
# Functions for training approaches - end
# ----------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #Create a parser object to collect the environment variables
    #that are in the default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load data from the location specified by args.train (In this case, an S3 bucket).
    kaggle_features = pd.read_csv(os.path.join(args.train, 'kaggle_features_small.csv'), index_col=0, engine="python")
    kaggle = pd.read_csv(os.path.join(args.train, 'kaggle_small.csv'), index_col=0, engine="python")
    onp = pd.read_csv(os.path.join(args.train, 'onp_small.csv'), index_col=0, engine="python")

    # Create a list of models to serve
    pipeline_list = []

    # Kaggle PAC pipeline
    pipeline_0 = kaggleFakeNewsClassifier(kaggle)
    pipeline_list.append(pipeline_0)

    # Kaggle/Popularity classifier
    pipeline_3 = kagglePopularityRecommender(kaggle, kaggle_features)
    pipeline_list.append(pipeline_3)

    # Kaggle/Popularity regressor
    # pipeline_3 = kagglePopularityRecommender(kaggle, kaggle_features)
    pipeline_list.append(pipeline_3)

    # Kaggle/Popularity Clustering pipeline
    # pipeline_3 = kagglePopularityRecommender(kaggle, kaggle_features)
    pipeline_list.append(pipeline_3)

    print('Appended to pipeline list: {}\nModel: {}'.format(pipeline_0, pipeline_0['model']))

    # Save the model_list to the location specified by args.model_dir
    joblib.dump(pipeline_list, os.path.join(args.model_dir, "pipeline_list.joblib"))


# ----------------------------------------------------------------------------------------------------------------------------
# Functions for invoke endpoint - start
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "pipeline_list.joblib"))


def input_fn(request_body, request_content_type):
    request_object = json.loads(request_body)

    return request_object

def predict_fn(input_object, pipeline_list):
    pipeline_index = input_object['pipeline_index']
    
    input_data = input_object['input_data']
    title = input_data['title']
    author = input_data['author']
    text = input_data['text']

    pipeline = pipeline_list[pipeline_index]
    model = pipeline['model']

    X_test = ''

    # Kaggle PAC classifier inference
    if pipeline_index==0:
        ps = PorterStemmer()
        tfidfVectorizer = pipeline['tfidf']

        title = re.sub('[^a-zA-z]', ' ', title)
        title = title.lower()
        title = title.split()
        title = [ps.stem(word) for word in title if not word in stopword]
        title = ' '.join(title)
        
        text = re.sub('[^a-zA-z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [ps.stem(word) for word in text if not word in stopword]
        text = ' '.join(text)

        X_test = tfidfVectorizer.transform([title]).toarray()

        return [float(p) for p in model.predict(X_test)]

    elif pipeline_index==1:
        return
    elif pipeline_index==2:
        return

    # Kaggle/Popularity clustering
    elif pipeline_index==3:
        X_test = pd.DataFrame(data=[[title, text]], columns=['title', 'text'])
        print('X_test: {}'.format(X_test))
        
        X_test = extractFeatures(X_test)        
        print('X_test: {}'.format(X_test))

        data_ref = pipeline['data_ref']

        predicted_cluster = model.predict(X_test)
        print('Predicted Cluster: {}'.format(predicted_cluster))
        dataset_cluster = data_ref[data_ref['clusters']==predicted_cluster]
        
        recommendation = dataset_cluster.sample(5)

        print('Recommendation: {}'.format(recommendation))
        print('Recommendation Numpy: {}'.format(recommendation.to_numpy()))

        return recommendation.to_numpy()

    return
    

def output_fn(prediction, content_type):
    print('Prediction: {}'.format(prediction))

    response_string = '{"prediction": []}'
    response_object = json.loads(response_string)

    for c in prediction:
        response_object['prediction'].append(c)

    return response_object

# Functions for invoke endpoint - end
# ----------------------------------------------------------------------------------------------------------------------------