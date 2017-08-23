import urllib.request

countries = "gb au at br ca de fr in it nl nz pl ru sg us za".split(" ")

categories = "scientific-qa-jobs consultancy-jobs pr-advertising-marketing-jobs engineering-jobs it-jobs accounting-finance-jobs".split(" ")

no_results = "\"results\":[]"

def request_page(country, category, page):
    # Credentials no longer valid
    r = urllib.request.urlopen("http://api.adzuna.com:80/v1/api/jobs/{}/search/{}?app_id=92cd0f19&app_key=8a9c57473fdeded0f90386b69569e1ba&results_per_page=50&max_days_old=20000&category={}&sort_direction=up&sort_by=date".format(country, page, category)).read().decode('utf-8')
    if (no_results in r):
        return 0
    else:
        file = open("Raw Data2/{}_{}_{}.json".format(country, category, page), "w")
        print(r, file=file)
        file.close()
        return 1

def request_adzuna(countries, categories):
    for country in countries:
        for category in categories:
            continue_requests = 1
            page_index = 1
            while (continue_requests == 1):
                print("Printing page {} of {} in {}.".format(page_index, category, country))
                continue_requests = request_page(country, category, page_index)
                page_index += continue_requests

request_adzuna(countries, categories)
