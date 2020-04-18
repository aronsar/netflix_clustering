from scipy.sparse.linalg import eigs
from csv import reader
from collections import defaultdict
import numpy as np
import time

# Notes:
# consider setting K_NEB to 24
# consider setting FEW_MOVIES_THRESHOLD to 10
# consider modifying similarity function to give lower scores to users who have fewer movies in common
    # or perhaps if their intersection over union is too low they should get penalized

LOAD_SIM_MAT = True
NUM_USERS = 5905 # counted
USERS_TO_USE = 100
NUM_MOVIES = 17770 # gotten from movie_titles.txt
K_NEB = 8 # value of k to use in k-nearest neighbor similarity graph
NUM_CLUSTERS = 30 # also the number of eigenvectors we compute
FEW_MOVIES_THRESHOLD = 3
FEW_MOVIES_IN_COMMON_DIFFERENCE = 10 # possible values range between 0 and 16, 16 being the most different

def dist_func(user_rating_vec1, user_rating_vec2):
    common_movies_idx = np.nonzero(user_rating_vec1 * user_rating_vec2)[0]
    # print(len(common_movies_idx))
    if len(common_movies_idx) < FEW_MOVIES_THRESHOLD:
        return FEW_MOVIES_IN_COMMON_DIFFERENCE / 16 # if two users have few movies in common they probably have different tastes

    common_movies_vec1 = user_rating_vec1[common_movies_idx]
    common_movies_vec2 = user_rating_vec2[common_movies_idx]
    op1 = common_movies_vec1 - common_movies_vec2
    op2 = np.square(op1)
    op3 = np.sum(op2)
    op4 = op3 / (16 * len(op1)) # 16 is the greatest possible squared rating difference
    return op4

       
if __name__ == '__main__':
    users = defaultdict(list) # dict: keyed on users, values are lists of the form [(movie_id, rating)...]
    uhash = {}
    hashnum = 0 # each user_id maps to a unique integer between 0 and 5904
    ratings = np.zeros([NUM_MOVIES, NUM_USERS])
    t = time.time() 
    with open('./data/train.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader) # skip first row
        for row in csv_reader:
            movie_id, user_id, rating, date = row
            if user_id not in uhash:
                uhash[user_id] = hashnum
                hashnum += 1
            users[uhash[user_id]].append((user_id, movie_id, rating))
            ratings[int(movie_id)-1, uhash[user_id]] = rating 
    
    if not LOAD_SIM_MAT:
        # define the distance matrix (dense matrix)
        distance_matrix = np.zeros([hashnum, hashnum])
        for row in range(hashnum):
            for col in range(row):
                # leaves diagonals as 0 FIXME: should users be similar to themselves?
                distance_matrix[row, col] = dist_func(ratings[:, row], ratings[:, col])
                distance_matrix[col, row] = distance_matrix[row, col]
        print(time.time() - t)

        # create similarity graph (sparse)
        similarity_matrix = 1 - distance_matrix

    else:
        import pickle
        with open('sim_mat.pkl', 'rb') as f:
            similarity_matrix = pickle.load(f)
    
    import pdb; pdb.set_trace()
    similarity_graph = np.zeros(np.shape(similarity_matrix))
    for user in range(hashnum):
        # find indices of k nearest neighbors (largest similarity values)
        k_neb_idx = np.argsort(similarity_matrix[user])[-K_NEB:]
        similarity_graph[user, :][k_neb_idx] = similarity_matrix[user][k_neb_idx]
        similarity_graph[:, user][k_neb_idx] = similarity_matrix[user][k_neb_idx]

    # create degree matrix
    degree_matrix = np.zeros(np.shape(similarity_matrix))
    for user in range(hashnum):
        degree_matrix[user, user] = np.sum(similarity_graph[user])

    # create unnormalized Laplacian
    unnormalized_laplacian = degree_matrix - similarity_graph
    
    # find smallest NUM_CLUSTERS eigenvectors of the unnormalized laplacian
    eigenvalues, eigenvectors = eigs(unnormalized_laplacian, k=NUM_CLUSTERS, which='SM')
    import pdb; pdb.set_trace()
