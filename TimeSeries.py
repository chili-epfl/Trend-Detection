"""Computes the distribution of the posting date of the job ads and outputs it in the Time Series folder"""

import os
import json

if __name__ == "__main__":

    # Where files are fetched
    DIRECTORY = 'Raw Data'

    # Saves relevant data here
    data_dict = {}

    # Loops over json files only
    for filename in os.listdir(DIRECTORY):
        if filename.endswith(".json"):

            # Job Category
            category = '_'.join(filename.split("_")[:2])

            # Initialises dict if not already done
            if category not in data_dict:
                data_dict[category] = {}

            # Gets json data from file
            with open(os.path.join(DIRECTORY, filename)) as f:
                json_data = json.load(f)["results"]

            # Fills counts in data dictionary, per category, year, month and day
            for key in range(len(json_data)):
                date = json_data[key]['created'][:10]
                year = int(date[3])
                month = int(date[5:7])
                day = int(date[8:])
                if year not in data_dict[category]:
                    data_dict[category][year] = {}
                    data_dict[category][year][0] = 0
                if month not in data_dict[category][year]:
                    data_dict[category][year][month] = {}
                    data_dict[category][year][month][0] = 0
                if day not in data_dict[category][year][month]:
                    data_dict[category][year][month][day] = 0
                data_dict[category][year][month][day] += 1
                data_dict[category][year][month][0] +=1
                data_dict[category][year][0] +=1

    # Prints data
    for category in data_dict:

        # Log
        print(category)

        # Printed File
        file = open("Time Series/{}.txt".format(category), "w")
        sorted(data_dict[category])
        for year in data_dict[category]:
            print("201{}\t{}".format(year, data_dict[category][year][0]), file=file)
            for month in range(1,13):
                if month in data_dict[category][year]:
                    for day in range(0,32):
                        if day in data_dict[category][year][month]:
                            print("201{}-{}-{}\t{}".format(year, month, day, data_dict[category][year][month][day]), file=file)
        file.close()