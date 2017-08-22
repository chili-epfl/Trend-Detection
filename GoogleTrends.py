from pytrends.request import TrendReq
import math
import os
import pandas as pd
import requests
import json
from TrendPositions import get_trending_words

pt = TrendReq('', '')
df_empty = pd.DataFrame({'A' : []})

### JSON ###

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
headers = {'User-Agent' : user_agent}

def get_interest_over_time_json(country, keyword):
    url = "https://trends.google.com/trends/fetchComponent?hl=en-US&q={}&geo={}&cid=TIMESERIES_GRAPH_0&export=3&w=500&h=300".format(keyword, country.upper())
    r = requests.get(url, headers=headers).text
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

def get_energy_json(df):
    energy = 0
    last_element = df[-1]
    interval_count = len(df) - 1
    for index in range(interval_count):
        energy += (math.pow(last_element, 2) - math.pow(df[index], 2)) * 1 / (interval_count - index)
    return energy

### HTML ###

def get_interest_over_time(pytrend, country, keyword, recursion):
    try:
        pytrend.build_payload([keyword], timeframe="2016-01-01 2017-06-30", geo=country.upper())
        try:
            return pytrend.interest_over_time()
        except:
            return df_empty
    except:
        if recursion <= 100:
            pt = TrendReq('', '')
            return get_interest_over_time(pt, country, keyword, recursion + 1)
        else:
            return df_empty

def get_energy(df):
    if df.empty:
        return "NaN"
    energy = 0
    rows = list(df.iterrows())
    last_element = rows[-1][1][0]
    interval_count = len(rows) - 1
    for index in range(interval_count):
        energy += (math.pow(last_element, 2) - math.pow(rows[index][1][0], 2)) * 1 / (interval_count - index)
    return energy

### CHOSE METHOD ###

def energy_method(method, pytrend, country, keyword, recursion):
    if method == "json":
        return get_interest_over_time_json(country, keyword)
    elif method == "html":
        interest = get_interest_over_time(pytrend, country, new_keyword, recursion)
        return get_energy(interest)
    return "NaN"

### MAIN ###

directory = "TF IDF Delta"
valid_dir = "Trend Positions"
dest_dir = "Google Trends"

processed_files = os.listdir(dest_dir)
valid_files = [filename for filename in os.listdir(valid_dir) if filename.endswith(".txt") and not filename.endswith("_no_nlp.txt") and filename not in processed_files]

for filename in valid_files:
    print(filename)
    keywords = get_trending_words(filename)
    file = open("{}/{}".format(dest_dir, filename), "w")
    for keyword in keywords:
        new_keyword = keyword.replace("_", " ")
        country = filename[:2]
        energy = energy_method("html", pt, country, new_keyword, 1)
        print(keyword, keywords[keyword], energy)
        print("{}\t{}\t{}".format(keyword, keywords[keyword], energy), file=file)
    file.close()