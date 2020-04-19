import random
import scipy
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
    action='store_true',
    help='Recalculate the similarity matrix from scratch') 
parser.add_argument(
    '-n',
    '--k_neb', 
    type=int,
    default=15,
    help='The number of nearest neighbors when constructing the similarity graph.') 
parser.add_argument(
    '-k',
    '--num_clusters', 
    type=int,
    default=10,
    help='The number of clusters to use with k-means.') 
parser.add_argument(
    '-u',
    '--users_to_use', 
    type=int,
    default=300,
    help='The number of users to use; fewer is faster; 5905 is maximum and takes >1 hr.') 
args = parser.parse_args()
assert(args.users_to_use <= 5905)


TRAIN_VAL_SPLIT = .9 # fraction of movies in train data for each user
NUM_USERS = 5905 # counted
NUM_MOVIES = 17770 # gotten from movie_titles.txt
NUM_EIGS = 30 # number of eigenvectors to compute
FOUND_COUNT = 0
NOTFOUND_COUNT = 0
def sim_func(user_rating_vec1, user_rating_vec2):
    common_movies_idx = np.nonzero(user_rating_vec1 * user_rating_vec2)[0]
    #either_movies_idx = np.nonzero(user_rating_vec1 + user_rating_vec2)[0]
    smaller_vec = min(len(np.nonzero(user_rating_vec1)[0]), len(np.nonzero(user_rating_vec2)[0]))

    common_movies_vec1 = user_rating_vec1[common_movies_idx]
    common_movies_vec2 = user_rating_vec2[common_movies_idx]
    op1 = common_movies_vec1 - common_movies_vec2
    op2 = np.square(op1)
    op3 = np.sum(op2)
    op4 = op3 / (16 * len(common_movies_idx) + .0001) # 16 is the greatest possible squared rating difference
    op5 = (1 - op4)  * ((len(common_movies_idx) / smaller_vec) ** .05) # hard to say how important intersection over union is...
    return op5


def unnormalized_spectral_clustering(ratings, args):
     # define the similarity matrix (dense matrix)
    if not args.load_sim_mat:
        t = time.time() 
        similarity_matrix = np.ones([args.users_to_use, args.users_to_use])
        for row in range(args.users_to_use):
            for col in range(row):
                # leaves diagonals as 1 FIXME: should users be similar to themselves?
                similarity_matrix[row, col] = sim_func(ratings[:, row], ratings[:, col])
                similarity_matrix[col, row] = similarity_matrix[row, col]
        with open('sim_mat.pkl', 'wb') as f:
            pickle.dump(similarity_matrix, f)
        print("Time taken to construct similarity matrix: {:.1f}".format(time.time() - t))


    else:
        with open('sim_mat.pkl', 'rb') as f:
            similarity_matrix = pickle.load(f)
    
    # create similarity graph (sparse)
    similarity_graph = np.zeros(np.shape(similarity_matrix))
    for user in range(args.users_to_use):
        # find indices of k nearest neighbors (largest similarity values)
        k_neb_idx = np.argsort(similarity_matrix[user])[-args.k_neb:]
        similarity_graph[user, :][k_neb_idx] = similarity_matrix[user][k_neb_idx]
        similarity_graph[:, user][k_neb_idx] = similarity_matrix[user][k_neb_idx]

    # create degree matrix
    degree_matrix = np.zeros(np.shape(similarity_graph))
    for user in range(args.users_to_use):
        degree_matrix[user, user] = np.sum(similarity_graph[user])

    # create unnormalized Laplacian
    unnormalized_laplacian = degree_matrix - similarity_graph
    assert(np.all(unnormalized_laplacian - unnormalized_laplacian.T == 0))
    np.linalg.cholesky(unnormalized_laplacian + np.eye(args.users_to_use)*.01) # test for positive semidefiniteness

    # find smallest NUM_EIGS eigenvalues of the unnormalized laplacian
    #eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(unnormalized_laplacian, k=NUM_EIGS, which='SM')
    eigenvalues, eigenvectors = np.linalg.eig(unnormalized_laplacian)
    smallest_idx = np.argsort(eigenvalues)[:NUM_EIGS]
    eigenvalues = eigenvalues[smallest_idx]
    eigenvectors = eigenvectors[:, smallest_idx]

    # create U, the matrix containing the eigenvectors as columns
    U = np.array(eigenvectors[:, 1:args.num_clusters+1])
    
    # let's do some k-means!
    kmeans = KMeans(n_clusters=args.num_clusters).fit_predict(U)
    return kmeans, similarity_graph


def create_train_cluster2users_dict(train_clusters):
    # given a cluster, we want a list of users in that cluster
    train_cluster2users_dict = defaultdict(list)
    for user, cluster in enumerate(train_clusters):
        train_cluster2users_dict[cluster].append(user)
    return train_cluster2users_dict


def predict_ratings(train_clusters, train_ratings_mat, val_users):
    # set the predicted rating of a user for a particular movie to the average rating of 
    # the cluster of users that the user belongs to
    global FOUND_COUNT, NOTFOUND_COUNT
    pred_val_ratings = defaultdict(list)
    train_cluster2users_dict = create_train_cluster2users_dict(train_clusters)

    for user in range(args.users_to_use):
        movie_id_arr = np.array(val_users[user])[:,1]
        for movie_id in movie_id_arr:
            peer_arr = train_cluster2users_dict[train_clusters[user]] # a peer is another in-cluster user
            peer_ratings = train_ratings_mat[movie_id, peer_arr]
            peer_ratings = peer_ratings[np.nonzero(peer_ratings)]
            if peer_ratings.size == 0:
                avg_peer_rating = 3 # FIXME: set this to the averge rating for that movie
                NOTFOUND_COUNT += 1
            else:
                avg_peer_rating = np.sum(peer_ratings) / len(peer_ratings)
                FOUND_COUNT += 1
            pred_val_ratings[user].append(int(avg_peer_rating + .5))

    return pred_val_ratings


def loss_func(pred_ratings, gt_ratings):
    loss = 0
    num_preds = 0
    for user in range(args.users_to_use):
        assert(len(pred_ratings[user]) == len(gt_ratings[user]))
        assert(not np.any(pred_ratings[user] == 0))
        assert(not np.any(gt_ratings[user] == 0))
        for pred_rating, gt_rating in zip(pred_ratings[user], gt_ratings[user]):
            loss += (pred_rating - gt_rating) ** 2
            num_preds += 1

    avg_error = loss / num_preds
    return loss, avg_error


def train_val_split(all_users, all_ratings):
    # training and validation data are split randomly
    train_users = {}
    val_users = {}
    train_ratings_mat = np.zeros(all_ratings.shape)
    val_ratings_mat = np.zeros(all_ratings.shape)
    
    for user in range(args.users_to_use):
        num_movies = len(all_users[user])
        num_train_movies = int(num_movies * TRAIN_VAL_SPLIT)
        train_movie_idx = random.sample(range(num_movies), num_train_movies)
        train_mask = np.zeros(num_movies, dtype=np.int)
        train_mask[train_movie_idx] = 1
        val_mask = np.ones(num_movies, dtype=np.int)
        val_mask[train_movie_idx] = 0
        train_users[user] = [all_users[user][i] for i in range(num_movies) if train_mask[i] == 1]
        val_users[user] = [all_users[user][i] for i in range(num_movies) if val_mask[i] == 1]

        train_movie_idxs = np.array(train_users[user])[:,1]
        train_ratings_mat[train_movie_idxs, user] = all_ratings[train_movie_idxs, user]
        val_movie_idxs = np.array(val_users[user])[:,1]
        val_ratings_mat[val_movie_idxs, user] = all_ratings[val_movie_idxs, user]
    
    return train_users, train_ratings_mat, val_users, val_ratings_mat


def inspect_similarity_graph(similarity_graph, all_users):
    movie_info = {}
    with open('./data/movie_titles.txt', 'r', encoding='ISO-8859-1') as f:
        header = next(f)
        for row in f:
            splitrow = row.rstrip().split(',')
            movie_id = splitrow[0]
            year_produced = splitrow[1]
            title = ''.join(splitrow[2:])
            movie_info[int(movie_id)-1] = (year_produced, title)

    for user in range(args.users_to_use):
        print('User {:d} has watched:'.format(user))
        for _, movie, rating in all_users[user][:15]:
            print('    {:d} --- {:s} --- {:s}'.format(rating, movie_info[movie][0], movie_info[movie][1]))

        peers = similarity_graph[user].nonzero()[0]
        for peer in peers:
            count = 0
            avg_rating_diff = 0
            print('Peer {:d} has these movies in common with the user:'.format(peer))
            for _, movie, rating in all_users[peer]:
                if movie in np.array(all_users[user])[:,1]:
                    idx = np.where(np.array(all_users[user])[:,1] == movie)[0][0]
                    user_rating = all_users[user][idx][2]
                    print('   {:d} / {:d} --- {:s} --- {:s}'.format(rating, user_rating, movie_info[movie][0], movie_info[movie][1]))
                    avg_rating_diff += (rating - user_rating) **2
                    count += 1
            print(avg_rating_diff / (count + .00001))


def gt_ratings(users):
    gt_ratings = defaultdict(list)
    for user in range(args.users_to_use):
        gt_ratings[user] = np.array(users[user])[:, 2]
    return gt_ratings

if __name__ == '__main__':
    all_users = defaultdict(list) # dict: keyed on users, values are lists of the form [(movie_id, rating)...]
    uhash = {}
    hashnum = 0 # each user_id maps to a unique integer between 0 and 5904
    all_ratings = np.zeros([NUM_MOVIES, NUM_USERS])

    with open('./data/train.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader) # skip first row
        for row in csv_reader:
            movie_id, user_id, rating, date = row
            if user_id not in uhash:
                uhash[user_id] = hashnum
                hashnum += 1
            all_users[uhash[user_id]].append((int(user_id), int(movie_id)-1, int(rating)))
            all_ratings[int(movie_id)-1, uhash[user_id]] = int(rating) 
   
    train_users, train_ratings_mat, val_users, val_ratings_mat = train_val_split(all_users, all_ratings)

    train_clusters, similarity_graph = unnormalized_spectral_clustering(train_ratings_mat, args)
    print("How many users in each cluster:")
    print([np.count_nonzero(train_clusters == i) for i in range(args.num_clusters)])
    #inspect_similarity_graph(similarity_graph, all_users)

    pred_val_ratings = predict_ratings(train_clusters, train_ratings_mat, val_users)
    loss, avg_error = loss_func(pred_val_ratings, gt_ratings(val_users))
    print("Loss is {:.1f}, and avg error is {:.2f}".format(loss, avg_error))
    print(FOUND_COUNT, NOTFOUND_COUNT)
    pass
