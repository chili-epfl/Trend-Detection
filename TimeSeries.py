import os
import json

directory = 'Raw Data'

map = {}

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        category = '_'.join(filename.split("_")[:2])
        if category not in map:
            map[category] = {}
        with open(os.path.join(directory, filename)) as f:
            data = json.load(f)["results"]
        for key in range(len(data)):
            date = data[key]['created'][:10]
            year = int(date[3])
            month = int(date[5:7])
            day = int(date[8:])
            if year not in map[category]:
                map[category][year] = {}
                map[category][year][0] = 0
            if month not in map[category][year]:
                map[category][year][month] = {}
                map[category][year][month][0] = 0
            if day not in map[category][year][month]:
                map[category][year][month][day] = 0
            map[category][year][month][day] += 1
            map[category][year][month][0] +=1
            map[category][year][0] +=1

for category in map:
    print(category)
    file = open("Time Series/{}.txt".format(category), "w")
    sorted(map[category])
    for year in map[category]:
        print("201{}\t{}".format(year, map[category][year][0]), file=file)
        for month in range(1,12):
            if month in map[category][year]:
                for day in range(0,31):
                    if day in map[category][year][month]:
                        print("201{}-{}-{}\t{}".format(year, month, day, map[category][year][month][day]), file=file)
    file.close()