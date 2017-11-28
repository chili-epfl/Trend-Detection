"""Generates json files, one per page, of the job ads of Adzuna in the Raw Data folder"""

import urllib.request

def request_page(country, category, page):
    """
    Requests a page from Adzuna, using an Adzuna-provided APP_ID and APP_KEY.
    All possible parameters are defined below.
    :param country: string, the country for which Adzuna searches
    :param category: string, the category of the jobs
    :param page: int, the number of the page
    :return: 0 if results are not found, 1 if found and a file is printed to keep the search results
    """
    # Credentials no longer valid
    r = urllib.request.urlopen("http://api.adzuna.com:80/v1/api/jobs/{}/search/{}?app_id={}&app_key={}&results_per_page=50&max_days_old=20000&category={}&sort_direction=up&sort_by=date".format(country, page, APP_ID, APP_KEY, category)).read().decode('utf-8')
    if (NO_RESULTS in r):
        return 0
    else:
        file = open("Raw Data/{}_{}_{}.json".format(country, category, page), "w")
        print(r, file=file)
        file.close()
        return 1

def request_adzuna(countries, categories):
    """
    Loops over the selected countries and categories by requesting all possible pages on Adzuna.
    :param countries: list of strings, countries over which the search loops
    :param categories: list of strings, categories over which the search loops
    :return: None, results are printed into json files, and progression logs in the console
    """
    for country in countries:
        for category in categories:
            continue_requests = 1
            page_index = 1
            while (continue_requests == 1):
                print("Printing page {} of {} in {}.".format(page_index, category, country))
                continue_requests = request_page(country, category, page_index)
                page_index += continue_requests

if __name__ == "__main__":

    # 2-letter codes of countries that we loop over, these are all the ones available in Adzuna
    COUNTRIES = "gb au at br ca de fr in it nl nz pl ru sg us za".split(" ")

    # Categories of jobs we will look for, not exhaustive list
    CATEGORIES = "scientific-qa-jobs consultancy-jobs pr-advertising-marketing-jobs engineering-jobs it-jobs accounting-finance-jobs".split(" ")

    # String found if search yields no results
    NO_RESULTS = "\"results\":[]"

    # Adzuna-provided app_id and app_key, no longer valid
    APP_ID = "92cd0f19"
    APP_KEY = "8a9c57473fdeded0f90386b69569e1ba"

    # Requests Adzuna to save jobs as json files for all categories in all countries
    request_adzuna(COUNTRIES, CATEGORIES)