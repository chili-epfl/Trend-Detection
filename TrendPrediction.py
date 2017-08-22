from KeywordPipeline import get_countries, cross_domain_filtering, get_pre_processed_entries, select_keywords, get_tfidf_map, desc_dir
from TrendPositions import get_trending_words
import numpy as np
import os

def get_features_and_labels(corpus, trending_words):
    labels = []
    features = []
    words, sorted_tfidf = select_keywords(corpus)
    tfidf_map = get_tfidf_map(sorted_tfidf)
    label_0 = False
    label_1 = False
    for text in corpus:
        for position in range(len(text)):
            word = text[position]
            features.append([position, tfidf_map[word], int('_' in word)])
            try:
                x = trending_words[word]
                label = 1
                label_1 = True
            except:
                label = 0
                label_0 = True
            labels.append(label)
    if label_0 and label_1:
        return features, labels
    else:
        return features, []

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

dest_dir = "Trend Prediction"

def build_model():
    countries = get_countries(desc_dir)
    keywords_per_country = np.load('keys.npy').item()
    for country in countries:
        print(country)
        categories = sorted([filename for filename in os.listdir(desc_dir) if filename.startswith(country) and filename.endswith(".csv")])
        if len([category for category in categories if "{}.txt".format(category[:-4]) not in os.listdir(dest_dir)]) > 0:
            # Cross-Domain Filtering
            to_delete = cross_domain_filtering(keywords_per_country[country])
            # Trend Prediction
            for filename in categories:
                file_path = os.path.join(desc_dir, filename)
                trending_words = get_trending_words(filename)
                if trending_words:
                    print(filename)
                    try:
                        descriptions_2017 = np.load("{}/descriptions_2017_{}.npy".format(dest_dir, filename[:-4])).item()
                    except:
                        descriptions_2017 = get_pre_processed_entries(file_path, to_delete, 6, 2017)
                        np.save("{}/descriptions_2017_{}.npy".format(dest_dir, filename[:-4]), descriptions_2017)
                    features, labels = get_features_and_labels(descriptions_2017, trending_words)
                    if labels:
                        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)
                        clf = svm.SVC(kernel='linear', C=1)
                        scores = cross_val_score(clf, X_train, y_train, cv=5)
                        file = open("{}/{}.txt".format(dest_dir, filename[:-4]), 'w')
                        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), file=file)
                        file.close()

build_model()