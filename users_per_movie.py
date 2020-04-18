# import train.csv
# read in data into dict keyed on customer_id
import numpy as np


from csv import reader
from collections import defaultdict
import matplotlib.pyplot as plt

movie_user_count = defaultdict(int)

with open('./data/train.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader) # skip first row
    for row in csv_reader:
        movie_id, user_id, rating, date = row
        movie_user_count[movie_id] += 1

# has NUM_MOVIES entries, each entry being the number of views a movie has had
views_per_movie = [] 

for _, views in movie_user_count.items():
    views_per_movie.append(views)

plt.subplot(211)
hist, bins, _ = plt.hist(views_per_movie, bins = 40)
plt.ylabel('num movies')
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
plt.subplot(212)
plt.hist(views_per_movie, bins=logbins)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('num movies')
plt.xlabel('times viewed')
plt.show()



