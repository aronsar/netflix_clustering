# import train.csv
# read in data into dict keyed on customer_id
import numpy as np


from csv import reader
from collections import defaultdict
import matplotlib.pyplot as plt

user_movie_count = defaultdict(int)

with open('./data/train.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader) # skip first row
    for row in csv_reader:
        movie_id, user_id, rating, date = row
        user_movie_count[user_id] += 1

# has NUM_USERS entries, each entry being the number of movies a user has watched
movies_viewed_list = [] 

for _, movies_viewed in user_movie_count.items():
    movies_viewed_list.append(movies_viewed)

plt.subplot(211)
hist, bins, _ = plt.hist(movies_viewed_list, bins = 40)
plt.ylabel('num users')
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
plt.subplot(212)
plt.hist(movies_viewed_list, bins=logbins)
plt.xscale('log')
#plt.yscale('log')
plt.ylabel('num users')
plt.xlabel('movie views')
plt.show()



