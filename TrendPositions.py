import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, sys, os
from KeywordPipeline import get_countries, cross_domain_filtering, get_pre_processed_entries
import numpy as np

def get_trending_words(filename):
    trend_dir = "TF IDF Delta"
    file_path = os.path.join(trend_dir, "{}.txt".format(filename[:-4]))
    with open(file_path, 'r') as file:
        lines = file.readlines()
    if len(lines) <= 3:
        return {}
    else:
        trending_NLP = {}
        for line in lines[3:]:
            elts = line[:-1].split("\t")
            if len(elts) == 2:
                delta = float(elts[1])
                if delta >= 0:
                    trending_NLP[elts[0]] = delta
        return trending_NLP

def get_trend_positions(descriptions, trending_words):
    max_len = 0
    for desc in descriptions:
        l = len(desc)
        if l > max_len:
            max_len = l
    positions_count = np.full(max_len, 0)
    for desc in descriptions:
        for word_index in range(len(desc)):
            try:
                word = trending_words[desc[word_index]]
                positions_count[word_index] += 1
            except:
                continue
    return positions_count

def get_string_of_array(array):
    return '\t'.join(list([str(x) for x in array]))

def get_trend_position_graphs():
    countries = get_countries(desc_dir)
    keywords_per_country = np.load('keys.npy').item()
    for country in countries:
        print(country)
        filenames = sorted([filename for filename in os.listdir(desc_dir) if filename.startswith(country) and filename.endswith(".csv")])
        to_delete = cross_domain_filtering(keywords_per_country[country])
        for filename in filenames:
            trending_NLP = get_trending_words(filename)
            if trending_NLP:
                print(filename)
                file_path = os.path.join(desc_dir, filename)
                try:
                    desc_2017 = np.load("Trend Prediction/descriptions_2017_{}.npy".format(filename[:-4])).item()
                except:
                    desc_2017 = get_pre_processed_entries(file_path, to_delete, 6, 2017)
                    np.save("Trend Prediction/descriptions_2017_{}.npy".format(filename[:-4]), desc_2017)
                positions_count = get_trend_positions(desc_2017, trending_NLP)
                plt.title(filename)
                plt.figure(figsize=(100, 40))
                plt.xlabel('Positions of Trending Words in Text')
                plt.ylabel('Frequency of Position')
                plt.plot(np.arange(len(positions_count)), positions_count, 'bo-')
                plt.savefig('{}/{}.png'.format(dest_dir, filename[:-4]))
                file = open("{}/{}.txt".format(dest_dir, filename[:-4]), "w")
                print(get_string_of_array(np.arange(len(positions_count))), file=file)
                print(get_string_of_array(positions_count), file=file)
                file.close()

if __name__ == "__main__":
    directory = "Raw Text"
    desc_dir = "Lemmatised Text"
    dest_dir = "Trend Positions"
    csv.field_size_limit(sys.maxsize)
    get_trend_position_graphs()