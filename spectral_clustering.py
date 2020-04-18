from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from csv import reader
from collections import defaultdict
import numpy as np
import time
import pickle
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument(
    '-l',
    '--load_sim_mat', 
    action='store_true') 
args = parser.parse_args()


NUM_USERS = 5905 # counted
USERS_TO_USE = 300
NUM_MOVIES = 17770 # gotten from movie_titles.txt
K_NEB = 20 # value of k to use in k-nearest neighbor similarity graph
NUM_CLUSTERS = 10 # the k in k-means
NUM_EIGS = 30 # number of eigenvectors to compute

def sim_func(user_rating_vec1, user_rating_vec2):
    common_movies_idx = np.nonzero(user_rating_vec1 * user_rating_vec2)[0]
    either_movies_idx = np.nonzero(user_rating_vec1 + user_rating_vec2)[0]
    intersection = len(common_movies_idx)
    union = len(either_movies_idx)

    common_movies_vec1 = user_rating_vec1[common_movies_idx]
    common_movies_vec2 = user_rating_vec2[common_movies_idx]
    op1 = common_movies_vec1 - common_movies_vec2
    op2 = np.square(op1)
    op3 = np.sum(op2)
    op4 = op3 / (16 * intersection + .0001) # 16 is the greatest possible squared rating difference
    op5 = (1 - op4)  * ((intersection / union) ** .05) # hard to say how important intersection over union is...
    return op5


def unnormalized_spectral_clustering(ratings, args):
     # define the similarity matrix (dense matrix)
    if not args.load_sim_mat:
        t = time.time() 
        similarity_matrix = np.ones([USERS_TO_USE, USERS_TO_USE])
        for row in range(USERS_TO_USE):
            for col in range(row):
                # leaves diagonals as 1 FIXME: should users be similar to themselves?
                similarity_matrix[row, col] = sim_func(ratings[:, row], ratings[:, col])
                similarity_matrix[col, row] = similarity_matrix[row, col]
        with open('sim_mat.pkl', 'wb') as f:
            pickle.dump(similarity_matrix, f)
        print(time.time() - t)


    else:
        with open('sim_mat.pkl', 'rb') as f:
            similarity_matrix = pickle.load(f)
    
    # create similarity graph (sparse)
    similarity_graph = np.zeros(np.shape(similarity_matrix))
    for user in range(USERS_TO_USE):
        # find indices of k nearest neighbors (largest similarity values)
        k_neb_idx = np.argsort(similarity_matrix[user])[-K_NEB:]
        similarity_graph[user, :][k_neb_idx] = similarity_matrix[user][k_neb_idx]
        similarity_graph[:, user][k_neb_idx] = similarity_matrix[user][k_neb_idx]

    # create degree matrix
    degree_matrix = np.zeros(np.shape(similarity_matrix))
    for user in range(USERS_TO_USE):
        degree_matrix[user, user] = np.sum(similarity_graph[user])

    # create unnormalized Laplacian
    unnormalized_laplacian = degree_matrix - similarity_graph
    assert(np.all(unnormalized_laplacian - unnormalized_laplacian.T == 0))
    # FIXME: also test for positive semidefiniteness

    # find smallest NUM_EIGS eigenvectors of the unnormalized laplacian
    eigenvalues, eigenvectors = eigs(unnormalized_laplacian, k=NUM_EIGS, which='SM')
    print(np.absolute(eigenvalues))

    # create U, the matrix containing the eigenvectors as columns
    U = np.absolute(np.array(eigenvectors[:, 1:NUM_CLUSTERS+1]))
    
    # let's do some k-means!
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit_predict(U)
    return kmeans


def create_train_cluster2users_dict(train_user2cluster_vec):
    train_cluster2users_dict = defaultdict(list)
    for user, cluster in enumerate(train_user2cluster_vec):
        train_cluster2users_dict[cluster].append(user)
    return train_cluster2users_dict


def predict_ratings(train_user2cluster_vec, ratings, user_movies):
    # set the predicted rating of a user for a particular movie to the average rating of 
    # the cluster of users that the user belongs to
    user_pred_ratings = defaultdict(list)
    train_cluster2users_dict = create_train_cluster2users_dict(user2cluster_mat)
    for user in range(USERS_TO_USE):
        movie_id_list = user_movies[user]
        for movie_id in movie_id_list:
            peer_ratings_sum = 0
            for peer in train_cluster2users_dict[user]:
                peer_rating_sum += ratings[peer, movie_id]
            avg_peer_rating = peer_rating_sum / len(train_cluster2users_dict[user])
            user_pred_ratings[user].append(avg_peer_rating)
    return user_pred_ratings


def loss_func(user_pred_ratings, user_gt_ratings):
    loss = 0
    for user in range(USERS_TO_USE):
        assert((user_pred_ratings[user]) == len(user_gt_ratings[user]))
        assert(np.any(user_pred_ratings == 0) is False)
        assert(np.any(user_gt_ratings == 0) is False)
        for pred_rank, gt_rank in zip(user_pred_ratings[user], user_gt_ratings[user]):
            loss += (pred_rank - gt_rank) ** 2
    return loss

if __name__ == '__main__':
    users = defaultdict(list) # dict: keyed on users, values are lists of the form [(movie_id, rating)...]
    uhash = {}
    hashnum = 0 # each user_id maps to a unique integer between 0 and 5904
    ratings = np.zeros([NUM_MOVIES, NUM_USERS])


    # read shit into train and val fuck i'm tired
    with open('./data/train.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader) # skip first row
        for row in csv_reader:
            movie_id, user_id, rating, date = row
            if user_id not in uhash:
                uhash[user_id] = hashnum
                hashnum += 1
            users[uhash[user_id]].append((user_id, int(movie_id)-1, rating))
            ratings[int(movie_id)-1, uhash[user_id]] = rating 
   
    train_user2cluster_vec = unnormalized_spectral_clustering(ratings, args)
    print("How many users in each cluster:")
    print([np.count_nonzero(train_user2cluster_vec == i) for i in range(NUM_CLUSTERS)])

    import pdb; pdb.set_trace()

    val_user_movies = defaultdict(list)
    val_user_gt_ratings = defaultdict(list)

    with open('./data/val.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            movie_id, user_id, rating, date = row
            val_user_movies[uhash[user_id]].append(int(movie_id)-1)
            val_user_gt_ratings[uhash[user_id]].append(rating)
    
    val_user_pred_ratings = predict_ratings(train_user2cluster_vec, ratings, val_user_movies)
    loss = loss_func(val_user_pred_ratings, val_user_gt_ratings)
    print(loss)
    import pdb; pdb.set_trace()
    pass
