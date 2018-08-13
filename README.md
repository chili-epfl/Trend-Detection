# Trend-Detection
Detecting Trends in Job Advertisements.

## Authors

[Khalil Mrini](https://www.linkedin.com/in/khalilmrini/), [Kshitij Sharma](https://scholar.google.ch/citations?user=Wr8pFkEAAAAJ&hl=en), [Pierre Dillenbourg](https://people.epfl.ch/cgi-bin/people?id=155704&op=bio&lang=en&cvlang=en)

[Paper available here](https://infoscience.epfl.ch/record/256472?&ln=en).

## Abstract

We present an automatic method for trend detection in job ads. From a job-posting website, we collect job ads from 16 countries and in 8 languages and 6 job domains. We pre-process them by removing stop words, lemmatising and performing cross-domain filtering. Then, we improve the vocabulary by forming n-grams and restrict it by filtering based on named-entity and part-of-speech tags. We split the job ads to compare two time periods: the first halves of 2016 and 2017. A trending word is defined as a word with a higher TF-IDF weight in 2017 than in 2016. The results obtained show a close correlation between the position of a word in its text and its trendiness regardless of country, language or job domain.

## Coding Format

**Language:** Python 3.

**Packages Used:** nltk, numpy, matplotlib, scipy, polyglot, pandas, pytrends, bs4, requests, urllib, pattern3, pymystem3.

## Python Files Description

The files are described hereafter in the order they should be used:

1. `AdzunaJobAdRetriever.py`: Generates json files, one per page, of the job ads of Adzuna in the Raw Data folder
2. `AdzunaJobDescriptionFetcher.py`: Fetches the descriptions if available from the original website and outputs them in the Raw Text folder
3. `TrendDetectionPipeline.py`: Performs all of the trend detection, with the help of the following files:
   * `TreeTagger.py`: Implements a tree tagger class for pre-processing
   * `SequenceMining.py`: Implements the Generalised Sequential Pattern (GSP) Algorithm
4. `TimeSeries.py`: Gives counts of the number of job ads collected over time
5. `TrendPositions.py`: Computes Trend Positions in the pre-processed text
6. `GoogleTrends.py`: Computes the *energy* of a trending word in Google Trends for comparison
