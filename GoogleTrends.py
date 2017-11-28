"""
Outputs Google Trends data in the following format:
- 1st column: word
- 2nd column: TF-IDF delta
- 3rd column: Google Trends energy
"""

from pytrends.request import TrendReq
import math
import os
import pandas as pd
import requests
import json
from TrendPositions import get_trending_words

### JSON ###

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
HEADERS = {'User-Agent' : USER_AGENT}

def get_interest_over_time_json(country, keyword):
    """
    Returns energy over time (int) after fetching Google Data
    :param country: string
    :param keyword: string
    :return: int
    """
    url = "https://trends.google.com/trends/fetchComponent?hl=en-US&q={}&geo={}&cid=TIMESERIES_GRAPH_0&export=3&w=500&h=300".format(keyword, country.upper())
    r = requests.get(url, headers=HEADERS).text
    r = r.split("google.visualization.Query.setResponse(")[1][:-2].replace("new Date", "\"").replace("),", ")\",")
    data = json.loads(r)
    if data["status"] == "error":
        return "NaN"
    numbers = []
    for row in data["table"]["rows"]:
        label = row["c"][0]["f"]
        if "2016" in label or "2017" in label:
            numbers.append(row["c"][1]["v"])
    return get_energy_json(numbers)

def get_energy_json(energy_list):
    """
    Gets energy based on list provided.
    :param energy_list: list of int
    :return: int
    """
    energy = 0
    last_element = energy_list[-1]
    interval_count = len(energy_list) - 1
    for index in range(interval_count):
        energy += (math.pow(last_element, 2) - math.pow(energy_list[index], 2)) * 1 / (interval_count - index)
    return energy

### HTML ###

def get_interest_over_time(pytrend, country, keyword, recursion):
    """
    Gets interest over time on Google Trends of a keyword. 
    Loops again with new pytrend object if request was rejected.
    Returns empty DataFrame if keyword does not have enough data on Google Trends.
    :param pytrend: pytrend object
    :param country: string
    :param keyword: string
    :param recursion: int
    :return: Pandas DataFrame
    """
    # timeframe="2016-01-01 2017-06-30"
    try:
        pytrend.build_payload([keyword], timeframe="all", geo=country.upper())
        try:
            return pytrend.interest_over_time()
        except:
            return EMPTY_DF
    except:
        if recursion <= 100:
            pt = TrendReq('', '')
            return get_interest_over_time(pt, country, keyword, recursion + 1)
        else:
            return EMPTY_DF

def get_energy(df):
    """
    Gets energy from Pandas DataFrame
    :param df: Pandas DataFrame
    :return: energy
    """
    if df.empty:
        return "NaN"
    energy = 0
    rows = list(df.iterrows())[-20:-2]
    last_element = rows[-1][1][0]
    interval_count = len(rows) - 1
    for index in range(interval_count):
        energy += (math.pow(last_element, 2) - math.pow(rows[index][1][0], 2)) * 1 / (interval_count - index)
    return energy

### CHOOSE METHOD ###

def energy_method(method, pytrend, country, keyword, recursion):
    """
    Chooses energy method.
    :param method: string, can be 'html' or 'json'
    :param pytrend: PyTrend object
    :param country: string
    :param keyword: string
    :param recursion: int, to keep control over recursion
    :return: energy (int) or 'NaN' if could not be fetched
    """
    if method == "json":
        return get_interest_over_time_json(country, keyword)
    elif method == "html":
        interest = get_interest_over_time(pytrend, country, new_keyword, recursion)
        return get_energy(interest)
    return "NaN"

### MAIN ###

# Destination Directory
DEST_DIR = "Google Trends"

# Processed Files, to pick up where the program left off if interrupted
PROCESSED_FILES = os.listdir(DEST_DIR)

# Valid file directory
valid_dir = "Trend Positions"

# Files that were not processed
valid_files = [filename for filename in os.listdir(valid_dir)
               if filename.endswith(".txt")
               and not filename.endswith("_no_nlp.txt")
               and filename not in PROCESSED_FILES]

# Initialising empty PyTrend request
pt = TrendReq('', '')

# DataFrame returned if request yields no results
EMPTY_DF = pd.DataFrame({'A' : []})

# Looping over the files that were pre-processed
for filename in valid_files:

    # Log
    print(filename)

    # Reads Keywords
    keywords = get_trending_words(filename)

    # Printing corresponding energy for each keyword
    file = open("{}/{}".format(DEST_DIR, filename), "w")
    for keyword in keywords:
        new_keyword = keyword.replace("_", " ")
        country = filename[:2]
        energy = energy_method("html", pt, country, new_keyword, 1)
        print(keyword, keywords[keyword], energy)
        print("{}\t{}\t{}".format(keyword, keywords[keyword], energy), file=file)
    file.close()