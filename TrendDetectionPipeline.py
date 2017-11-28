"""
Trend Detection pipeline
- Outputs in Lemmatised Text, TF IDF Delta folders
"""

import csv
import sys
from nltk.tokenize import RegexpTokenizer
import os
from nltk.corpus import stopwords
from collections import Counter
import math
import operator
import re
from pprint import pprint
import numpy as np
from SequenceMining import GspSearch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

### LANGUAGE RECOGNITION ###

if __name__ == '__main__':
    # These are the available languages with stopwords from NLTK
    languages = stopwords.fileids()

    # Fill the dictionary of languages, to avoid  unnecessary function calls
    print("Loading stop words...", end='\r')
    try:
        dict_list = np.load('stopwords.npy').item()
    except:
        dict_list = {}
        for lang in languages:
            dict_list[lang] = {}
            for stop_word in stopwords.words(lang):
                dict_list[lang][stop_word] = 0
        np.save('stopwords.npy', dict_list)
    print("Loaded stop words.      ")

def test_for_language(descriptions):
    """
    Tests which language descriptions are written in.
    :param descriptions: list of strings
    :return: string
    """
    tokens = [item for d in descriptions for item in d]
    lang = which_language(tokens)
    return lang

def score(tokens):
    """
    Get text, find most common words and compare with known
    stopwords. Return dictionary of values
    :param tokens: list of strings, token
    :return: dict, score per language
    """
    # Evaluating scores for each language
    scorelist = {}
    for lang in dict_list:
        scorelist[lang] = 0
        for word in tokens:
            if lang in dict_list:
                if word in dict_list[lang]:
                    scorelist[lang] += 1
    return scorelist

def which_language(text):
    """
    This function just returns the language name, from a given
    "scorelist" dictionary as defined above.
    :param text: list of strings
    :return: string
    """
    scorelist = score(text)
    sorted_scorelist = sorted(scorelist.items(), key=operator.itemgetter(1))
    maximum = sorted_scorelist[-1][1]
    # Default Language is English
    if maximum == 0:
        return "english"
    return sorted_scorelist[-1][0]

### LEMMATISING ###

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from pattern3.fr import parse as frparse
from pattern3.nl import parse as nlparse
from pattern3.de import parse as deparse
from pattern3.it import parse as itparse
from pydic import PyDic
from pymystem3 import Mystem

if __name__ == "__main__":
    # Initialising Lemmatisers with logs
    print("Initialising lemmatiser for Polish...  ",end='\r')
    pl_dict = PyDic('pydic/odm.txt')
    print("Initialising lemmatiser for Russian... ",end='\r')
    ru_lemmatiser = Mystem()
    print("Initialising lemmatiser for English... ",end='\r')
    en_lemmatiser = WordNetLemmatizer()
    print("Done initialising lemmatisers.         ")

def pl_lemmatise(word):
    """
    Lemmatiser for Polish
    :param word: string
    :return: string
    """
    word_forms = pl_dict.word_base(word)
    if word_forms:
        return word_forms[0]
    return word

def ru_lemmatise(word):
    """
    Lemmatiser for Russian
    :param word: string
    :return: string
    """
    return ru_lemmatiser.lemmatize(word)[0]

def en_lemmatise(word):
    """
    Lemmatiser for English
    :param word: string
    :return: string
    """
    new_word = en_lemmatiser.lemmatize(word, wn.NOUN)
    if new_word == word:
        new_word = en_lemmatiser.lemmatize(word, wn.VERB)
        if new_word == word:
            new_word = en_lemmatiser.lemmatize(word, wn.ADJ)
            if new_word == word:
                new_word = en_lemmatiser.lemmatize(word, wn.ADV)
    return new_word

def lemmatise(word, language):
    """
    Redirects to language's lemmatiser.
    :param word: string
    :param language: string
    :return: string
    """
    try:
        return {
            'english': en_lemmatise(word),
            'french': frparse(word, lemmata=True).split('/')[-1],
            'dutch': nlparse(word, lemmata=True).split('/')[-1],
            'german': deparse(word, lemmata=True).split('/')[-1],
            'italian': itparse(word, lemmata=True).split('/')[-1],
            'polish': pl_lemmatise(word),
            'russian': ru_lemmatise(word)
        }[language]
    except:
        return word

### TEXT PRE-PROCESSING FOR LEMMATISATION ###

# Tokenizing, lemmatising, removing stop words, and patterns of format (hex color, digit-only or digit-started strings)

def get_entries(directory, filename):
    """
    Gets descriptions from file.
    :param directory: string
    :param filename: string
    :return: list of strings
    """
    lemmatised_files = os.listdir(DESC_DIR)
    if filename in lemmatised_files:
        with open('{}/{}'.format(DESC_DIR, filename), 'r') as csvfile:
            reader = list(csv.reader(csvfile))
            del reader[0]
            descriptions = [x[4].split(' ') for x in reader]
            return descriptions
    else:
        with open(os.path.join(directory, filename), 'r') as csvfile:
            # Reading file, deleting header, getting descriptions
            reader = list(csv.reader(csvfile))
            with open('{}/{}'.format(DESC_DIR, filename), 'w') as new_csvfile:
                header = reader[0]
                writer = csv.DictWriter(new_csvfile, fieldnames=header)
                writer.writeheader()
                del reader[0]
                descriptions = [tokenize(x[4]) for x in reader]
                # Language recognition and selection of corresponding stop word corpus
                lang = test_for_language(descriptions)
                # Pre-processing
                new_descriptions = []
                index = 0
                length = len(descriptions)
                for description in descriptions:
                    new_description = pre_process(description, lang)
                    new_descriptions.append(new_description)
                    writer.writerow({header[0]: reader[index][0], header[1]: reader[index][1], header[2]: reader[index][2], header[3]: reader[index][3], header[4]: ' '.join(new_description)})
                    index += 1
                    print("--> Lemmatising: {}/{}     ".format(index, length),end='\r')
                print("--> Done pre-processing for {} job ads.                    ".format(length))
                return new_descriptions

def tokenize(description):
    """
    Tokenizer
    :param description: string
    :return: list of strings
    """
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(description.lower())

def matches(pattern, string):
    """
    Checks if match pattern was found in string
    :param pattern: Regex
    :param string: string
    :return: Boolean
    """
    re_match = pattern.match(string)
    if bool(re_match):
        re_span = re_match.span()
        return re_span[0] == 0 and re_span[1] == len(string)
    return False

def pre_process(tokens, lang):
    """
    Pre_processes tokens for given language
    :param tokens: list of strings
    :param lang: string
    :return: list of strings
    """
    pattern_1 = re.compile(r"\d+([a-z]+)?")
    pattern_2 = re.compile(r"([a-f]|(\d)){6}")
    # Filtering Stop Words and details of words
    new_tokens = []
    for token in tokens:
        if lang not in dict_list or token not in dict_list[lang]:
            if not matches(pattern_1, token) and not matches(pattern_2, token) and '_' not in token and len(token) <= 20:
                new_token = lemmatise(token, lang)
                new_tokens.append(new_token)
    return new_tokens

### TEXT PRE PROCESSING AFTER LEMMATISING ###

# First creating N-GRAMS
# Removing words that cross-domain filtering detected,
# removing locations and persons using NER taggers,
# removing anything that is not nouns/unknown/adjectives using PoS tags

from TreeTagger import TreeTagger

def get_pre_processed_entries(filename, to_delete, min_month, max_month, year):
    """
    Pre-processing function
    Arguments:
        filename: path to the file to be read
        to_delete: map of the words that should be cross-domain-filtered
        min_month and max_month: we want job ads that were published from month with number min_month to month with number max_month
        year: we want job ads in this year
    """
    with open(filename, 'r') as csvfile:
        # Reading file, deleting header, getting descriptions
        reader = list(csv.reader(csvfile))
        del reader[0]
        # Language Detection
        lang = get_language(reader)
        try:
            tt = TreeTagger(language=lang)
        except:
            print("THE REJECTED LANGUAGE", lang)
            tt = TreeTagger(language='english')
        # Descriptions
        descriptions = [tokenize(x[4]) for x in reader if check_time(x[2], min_month, max_month, year)]
        descriptions = gsp_search(descriptions, 0.2)
        descriptions = [remove_words(prune_ner_tags(description), to_delete, tt) for description in descriptions]
        return descriptions

def check_time(string, min_month, max_month, year):
    return int(string[:4]) == year and int(string[5:7]) <= max_month and int(string[5:7]) >= min_month

### POS TAG AND CROSS DOMAIN FILTERING

from polyglot.detect import Detector

def get_language(reader):
    """
    Gets language of descriptions.
    :param reader: CSV file content
    :return: string
    """
    desc = [x[4] for x in reader]
    text = ' '.join(desc)
    try:
        lang = Detector(text).language.name
        if lang == "un":
            return test_for_language(desc)
        else:
            return lang.lower()
    except:
        return test_for_language(desc)

def remove_words(description, to_delete, tt):
    """
    Removes words from description
    :param description: list of strings, tokenized description
    :param to_delete: list of strings, words to delete
    :param tt: TreeTagger
    :return:
    """
    new_description = []
    for word in description:
        if word not in to_delete:
            if tt.is_acceptable_word(word):
                new_description.append(word)
    return new_description

### NER TAG PRUNING ###

from polyglot.text import Text

def prune_ner_tags(tokens):
    """
    Removes locations and persons as detected by NER tagger.
    :param tokens: list of strings
    :return: pruned list of strings
    """
    try:
        entities = set([location for entity in Text(' '.join(tokens)).entities for location in list(entity) if entity.tag in ["I-LOC", "I-PER"]])
        new_description = [token for token in tokens if token not in entities]
        return new_description
    except:
        return tokens

### SELECTION OF FREQUENT TERMS ###

def tf_idf(corpus):
    """
    Computes TF-IDF
    :param corpus: list of lists of strings
    :return: dict with TF-IDF score
    """
    df = {}
    tf = {}
    doc_len = len(corpus)
    for text in corpus:
        word_set = set(text)
        # Computing df
        for word in word_set:
            try:
                df[word] += 1
            except:
                df[word] = 1
        # Computing tf
        counter = Counter(text)
        try:
            tf_max = counter.most_common(1)[0][1]
        except:
            tf_max = len(text)
        for word in word_set:
            try:
                tf[word] += counter[word]/tf_max
            except:
                tf[word] = counter[word]/tf_max
    # Computing TF-IDF
    tfidf = {}
    for word in df:
        tfidf[word] = (math.log(tf[word]/doc_len)+1)*math.log(doc_len/df[word])
    return tfidf

def get_keywords_per_document(tfidf):
    """
    Gets highest-scoring words in TF IDF.
    :param tfidf: dict
    :return: ranking (dict) and sorted words by score (list)
    """
    sorted_tfidf = sorted(tfidf.items(), key=operator.itemgetter(1))
    ranking = {}
    word_count = len(sorted_tfidf)
    for x in range(word_count):
        ranking[sorted_tfidf[x][0]] = word_count - x
    return ranking, sorted_tfidf

def select_keywords(corpus):
    """
    Selects keywords using the above functions.
    :param corpus: list of lists of strings
    :return: ranking (dict) and sorted words by score (list)
    """
    tfidf = tf_idf(corpus)
    keywords, sorted_tfidf = get_keywords_per_document(tfidf)
    return keywords, sorted_tfidf

### CROSS-DOMAIN FILTERING ###

def cross_domain_filtering(keywords_per_country):
    """
    Cross-domain Filtering function
    :param keywords_per_country: dict
    :return: list of strings
    """
    keywords = set([keyword for category in keywords_per_country for keyword in category])
    threshold = min(len(keywords)*0.1, 1000)
    to_delete = {}
    for keyword in keywords:
        ranking = []
        for category in keywords_per_country:
            try:
                ranking.append(category[keyword])
            except:
                continue
        if len(ranking) >= 4 and np.std(ranking) <= threshold:
            to_delete[keyword] = 0
    return to_delete

### GSP SEARCH ###

def introduce_n_grams(text, n_grams):
    """
    Joins n-grams in text
    :param text: list of strings
    :param n_grams: list of strings
    :return: list of strings
    """
    new_text = ' '.join(text)
    for n_gram in n_grams:
        if n_gram in new_text:
            new_text = new_text.replace(n_gram, n_gram.replace(' ', '_'))
    return new_text.split(' ')

def gsp_search(descriptions, threshold):
    """
    Performs Generalised Sequential Pattern (GSP) search given threshold on descriptions
    :param descriptions: list of strings
    :param threshold: float
    :return: list of strings
    """
    gsp = GspSearch(descriptions)
    n_grams = gsp.search(threshold)
    new_descriptions = [introduce_n_grams(description, n_grams) for description in descriptions]
    return new_descriptions

### PIPELINE ###

def get_tfidf_for_trends(d1, d2):
    """
    Gets keywords using TF-IDF for 2 time periods
    :param d1: list of list of strings
    :param d2: list of list of strings
    :return: tuple of lists of strings
    """
    keywords_2016, tfidf_2016 = select_keywords(d1)
    keywords_2017, tfidf_2017 = select_keywords(d2)
    return tfidf_2016, tfidf_2017

def extract_text_for_trends(filename, to_delete):
    """
    Gets pre-processed descriptions
    :param filename: string
    :param to_delete: list of strings, used for Cross-Domain Filtering
    :return: tuple of lists of strings
    """
    file_path = os.path.join(DESC_DIR, filename)
    descriptions_2016 = get_pre_processed_entries(file_path, to_delete, 1, 6, 2016)
    descriptions_2017 = get_pre_processed_entries(file_path, to_delete, 1, 6, 2017)
    return descriptions_2016, descriptions_2017

def extract_keywords(directory, filename):
    """
    Gets keywords for a given file
    :param directory: string
    :param filename: string
    :return:
    """
    descriptions = get_entries(directory, filename)
    keywords, sorted_tfidf = select_keywords(descriptions)
    return keywords

### TREND DETECTION ###

def get_tfidf_map(tfidf):
    """
    Gets TF-IDF dict from TF-IDF tuple
    :param tfidf: list of tuples (string, float)
    :return: dict {string: float}
    """
    tfidf_map = {}
    for entry in tfidf:
        tfidf_map[entry[0]] = entry[1]
    return tfidf_map

def compare_tfidfs(tfidf_2016, tfidf_2017):
    """
    Compares two TF-IDF lists
    :param tfidf_2016: list of tuples (string, float)
    :param tfidf_2017: list of tuples (string, float)
    :return: sorted list of tuples (string, float)
    """
    tfidf_map_2016 = get_tfidf_map(tfidf_2016)
    tfidf_map_2017 = get_tfidf_map(tfidf_2017)
    tfidf_delta = {}
    for keyword in tfidf_map_2016:
        try:
            score_2017 = tfidf_map_2017[keyword]
            score_2016 = tfidf_map_2016[keyword]
            delta = score_2017 - score_2016
            tfidf_delta[keyword] = delta
        except:
            continue
    sorted_tfidf_delta = sorted(tfidf_delta.items(), key=operator.itemgetter(1))
    return sorted_tfidf_delta

### DENDROGRAM OF TRENDING TERMS ###

def compute_tfidf_vectors(corpus, trending_words):
    """
    Computes TF IDF vectors
    :param corpus: list of lists of strings
    :param trending_words: list of strings
    :return: dictionary {int: float}
    """
    text_count = len(corpus)
    word_count = len(trending_words)
    # Initialising DF
    df = np.zeros(word_count)
    tfidf = np.zeros((word_count, text_count))
    pprint(trending_words)
    for text_index in range(text_count):
        text = corpus[text_index]
        counter = Counter(text)
        try:
            tf_max = counter.most_common(1)[0][1]
        except:
            tf_max = len(text)
        for word_index in range(word_count):
            count = counter[trending_words[word_index]]
            if count > 0:
                # Computing df
                df[word_index] += 1
                # Computing tf
                tfidf[word_index][text_index] = count/tf_max
    # Computing TF-IDF
    for word_index in range(word_count):
        tfidf[word_index] = np.multiply(np.log(np.add(tfidf[word_index],1)), math.log(text_count/df[word_index]))
    return tfidf

### MAIN FUNCTIONS ###

def get_countries(directory):
    """
    Gets countries available in directory
    :param directory: string
    :return: list of strings
    """
    countries = set()
    for filename in os.listdir(directory):
        if not filename.startswith("._") and filename.endswith(".csv"):
            countries.add(filename[:2])
    return countries

def get_keywords_per_country(directory, dest_dir):
    """
    Gets keywords for each country
    :param directory: string
    :param dest_dir: string
    :return: None, prints into files
    """
    countries = get_countries(directory)
    # Keyword Extraction
    try:
        keywords_per_country = np.load('keys.npy').item()
    except:
        keywords_per_country = {}
        for country in countries:
            keywords_per_country[country] = []
        processed_files = os.listdir(dest_dir)
        for filename in os.listdir(directory):
            if not filename.startswith("._") and filename.endswith(".csv") and "{}.txt".format(filename[:-4]) not in processed_files:
                print("File: {}".format(filename))
                keywords = extract_keywords(directory, filename)
                country = filename[:2]
                keywords_per_country[country].append(keywords)
        np.save("keys.npy", keywords_per_country)
    for country in countries:
        print(country)
        categories = sorted([filename for filename in os.listdir(directory) if filename.startswith(country) and filename.endswith(".csv")])
        if len([category for category in categories if "{}.txt".format(category[:-4]) not in os.listdir(dest_dir)]) > 0:
            # Cross-Domain Filtering
            to_delete = cross_domain_filtering(keywords_per_country[country])
            # Trend Detection
            for index in range(len(categories)):
                filename = categories[index]
                if "{}.txt".format(filename[:-4]) not in os.listdir(dest_dir):
                    desc_2016, desc_2017 = extract_text_for_trends(filename, to_delete)
                    print(desc_2016)
                    print(desc_2017)
                    tfidf_2016, tfidf_2017 = get_tfidf_for_trends(desc_2016, desc_2017)
                    sorted_tfidf_delta = compare_tfidfs(tfidf_2016, tfidf_2017)
                    trending_words = []
                    file = open("{}/{}.txt".format(dest_dir, filename[:-4]), "w")
                    print("Words in 2016: {}".format(len(tfidf_2016)), file=file)
                    print("Words in 2017: {}".format(len(tfidf_2017)), file=file)
                    print("Common words: {}".format(len(sorted_tfidf_delta)), file=file)
                    for keyword in sorted_tfidf_delta:
                        print("{}\t{}".format(keyword[0], keyword[1]), file=file)
                        if keyword[1] > 0:
                            trending_words.append(keyword[0])
                    file.close()
                    # Building dendrogram
                    if trending_words:
                        try:
                            trending_tfidf_vectors = compute_tfidf_vectors(desc_2016 + desc_2017, trending_words)
                            plt.figure(figsize=(100, 40))
                            plt.title(filename[:-4], fontsize=13)
                            plt.xlabel('Trending Words', fontsize=10)
                            plt.ylabel('Cosine Distance of TF-IDF vectors', fontsize=10)
                            distances = linkage(trending_tfidf_vectors, 'average')
                            dendrogram(distances, labels=trending_words, leaf_font_size=8)
                            plt.savefig('{}/{}.png'.format(dest_dir, filename[:-4]))
                        except:
                            continue

### MAIN ###

if __name__ == '__main__':
    DESC_DIR = "Lemmatised Text"
    DIRECTORY = "Raw Text"
    DEST_DIR = "TF IDF Delta"
    csv.field_size_limit(sys.maxsize)
    get_keywords_per_country(DIRECTORY, DEST_DIR)