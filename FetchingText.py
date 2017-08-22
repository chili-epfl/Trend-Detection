import os
import json
import urllib.request as url
from bs4 import *

import csv

map = {}

old_category = ""

printed_out_categories = {}

for filename in os.listdir('Raw Text'):
    printed_out_categories[filename.split('.')[0]] = 0

directory = 'Raw Data'

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        print(filename)
        category = '_'.join(filename.split("_")[:2])
        try:
            x = printed_out_categories[category]
        except:
            if len(old_category) == 0:
                old_category = category
            if old_category != category:
                print("Printing the CSV file of {}".format(old_category))
                with open('Raw Text/{}.csv'.format(old_category), 'w') as csvfile:
                    fieldnames = ['id', 'title', 'created', 'description_from_url', 'description']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for job in map[old_category]:
                        writer.writerow(job)
                old_category = category
            try:
                x = map[category]
            except:
                map[category] = []
                continue
            with open(os.path.join(directory, filename)) as f:
                data = json.load(f)["results"]
            for key in range(len(data)):
                print("{} {}".format(filename, key))
                job = {}
                job['id'] = data[key]['id']
                job['title'] = data[key]['title']
                job['created'] = data[key]['created']
                description = data[key]['description']
                desc_from_url = -1
                try:
                    redirect_url = BeautifulSoup(url.urlopen(data[key]['redirect_url']).read())
                    if len(redirect_url('a')) == 1:
                        job_url = str(redirect_url('a')[0]).split("href=\"")[1].split("\">")[0]
                        try:
                            soup = BeautifulSoup(url.urlopen(job_url).read())
                            [s.extract() for s in soup('script')]
                            [s.extract() for s in soup('a')]
                            text = soup.get_text()
                            if description[:10] in text:
                                description = text[text.index(description[:10]):]
                                desc_from_url = 1
                                print("Description from URL")
                            else:
                                desc_from_url = 0
                                print("Redirect Target URL doesn't have Description")
                        except:
                            print("Could not get Redirect Target URL")
                            continue
                except:
                    print("Could not get Redirect Source URL")
                    continue
                job['description_from_url'] = desc_from_url
                job['description'] = description
                map[category].append(job)
            continue

