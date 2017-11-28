"""Fetch the descriptions if available from the original website and outputs them in the Raw Text folder"""

import os
import json
import urllib.request as url
from bs4 import *

import csv

if __name__ == "__main__":
    
    # Dictionary collecting all data
    data_dict = dict()

    old_category = None

    # Folder where the json files containing Adzuna search results are found
    JOB_AD_DIR = 'Raw Data'

    # Folder where the job ad descriptions will be saved
    DESCRIPTION_DIR = 'Raw Text'

    # If the Job Ad Description Fetcher was interrupted, this enables it to restart where it left off
    printed_out_categories = set(filename.split('.')[0] for filename in os.listdir(DESCRIPTION_DIR))
    
    # File Extensions
    JOB_AD_EXT = '.json'
    DESCRIPTION_EXT = '.csv'

    # CSV Header
    ID = 'id'
    TITLE = 'title'
    DATE = 'created'
    DESC_STATE = 'description_from_url'
    DESC = 'description'
    FIELDNAMES = [ID, TITLE, DATE, DESC_STATE, DESC]
    
    # Filenames to loop over
    job_ad_file_names = [filename for filename in os.listdir(JOB_AD_DIR) if filename.endswith(JOB_AD_EXT)]
    
    for filename in job_ad_file_names:

        # Log
        print(filename)
        
        # Category of file
        category = '_'.join(filename.split("_")[:2])
        
        # Proceeding if category was not already finished
        if category not in printed_out_categories:

            # Initialising old_category if None
            if old_category == None:
                old_category = category

            # If we change category, corresponding CSV file is printed
            if not old_category == category:
                print("Printing the CSV file of {}".format(old_category))
                with open('{}/{}{}'.format(DESCRIPTION_DIR, old_category, DESCRIPTION_EXT), 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
                    writer.writeheader()
                    for job in data_dict[old_category]:
                        writer.writerow(job)
                old_category = category

            # Initialises empty array in the data dictionary
            if category not in data_dict:
                data_dict[category] = []

            # Loading json data as dictionary
            with open(os.path.join(JOB_AD_DIR, filename)) as f:
                json_data = json.load(f)["results"]
                
            # Looping over json data
            for key in range(len(json_data)):

                # Log
                print("{} {}".format(filename, key))

                # Job Data
                job = dict()
                job[ID] = json_data[key][ID]
                job[TITLE] = json_data[key][TITLE]
                job[DATE] = json_data[key][DATE]
                description = json_data[key][DESC]

                # Checking state of description
                desc_from_url = -1

                # Trying to get description from Redirect Source URL
                try:
                    redirect_url = BeautifulSoup(url.urlopen(json_data[key]['redirect_url']).read())

                    # Determining if Redirect URL is in Adzuna or not
                    if len(redirect_url('a')) == 1:
                        job_url = str(redirect_url('a')[0]).split("href=\"")[1].split("\">")[0]
                        soup = BeautifulSoup(url.urlopen(job_url).read())
                        print("This is a redirect URL")
                    else:
                        soup = redirect_url
                        print("The Job Ad Description will be retrieved from Adzuna")

                    # Trying to get description from Redirect Target URL
                    try:
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
                        pass
                except:
                    print("Could not get Redirect Source URL")
                    pass
                job[DESC_STATE] = desc_from_url
                job[DESC] = description
                data_dict[category].append(job)

    # Printing last category
    try:
        print("Printing the CSV file of {}".format(old_category))
        with open('{}/{}{}'.format(DESCRIPTION_DIR, old_category, DESCRIPTION_EXT), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writeheader()
            for job in data_dict[category]:
                writer.writerow(job)
    except:
         print("Last category was empty")
         pass