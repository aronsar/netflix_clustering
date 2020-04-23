import random
import scipy
from csv import reader
from collections import defaultdict
import numpy as np
import time
import pickle
import argparse 
import torch

parser = argparse.ArgumentParser() 
parser.add_argument(
    '-t',
    '--latent_vec_size', 
    type=int,
    default=10,
    help='The length of the latent vector inherent to each user and movie.') 
parser.add_argument(
    '-p',
    '--load_params', 
    action='store_true',
    help='Load the trained parameters.') 
parser.add_argument(
    '-l',
    '--learning_rate', 
    type=float,
    default=0.003,
    help='Learning rate for optimizer') 
parser.add_argument(
    '-n',
    '--num_epochs', 
    type=int,
    default=5000,
    help='The number of steps to run the optimizer with') 
parser.add_argument(
    '-r',
    '--regularization', 
    type=float,
    default=.0001,
    help='The relative importance of the regularization loss.') 
args = parser.parse_args()


TRAIN_VAL_SPLIT = 1 # fraction of movies in train data for each user
NUM_USERS = 5905 # counted
NUM_MOVIES = 17770 # gotten from movie_titles.txt


def train_val_split(all_users, all_ratings):
    # training and validation data are split randomly
    train_users = {}
    val_users = {}
    train_ratings_mat = np.zeros(all_ratings.shape)
    val_ratings_mat = np.zeros(all_ratings.shape)
    
    for user in range(NUM_USERS):
        num_movies = len(all_users[user])
        num_train_movies = int(num_movies * TRAIN_VAL_SPLIT + .5)
        train_movie_idx = random.sample(range(num_movies), num_train_movies)
        train_mask = np.zeros(num_movies, dtype=np.int)
        train_mask[train_movie_idx] = 1
        val_mask = np.ones(num_movies, dtype=np.int)
        val_mask[train_movie_idx] = 0
        train_users[user] = [all_users[user][i] for i in range(num_movies) if train_mask[i] == 1]
        val_users[user] = [all_users[user][i] for i in range(num_movies) if val_mask[i] == 1]
        
        if len(np.shape(np.array(train_users[user]))) != 2:
            import pdb; pdb.set_trace()
        train_movie_idxs = np.array(train_users[user])[:,1]
        train_ratings_mat[train_movie_idxs, user] = all_ratings[train_movie_idxs, user]
        val_movie_idxs = np.array(val_users[user])
        if len(np.shape(np.array(val_users[user]))) == 2: # only relevant for single movie users
            val_movie_idxs = np.array(val_users[user])[:,1]
            val_ratings_mat[val_movie_idxs, user] = all_ratings[val_movie_idxs, user]
    
    return train_users, train_ratings_mat, val_users, val_ratings_mat


def avg_error_fn(val_ratings_mat, pred_ratings_mat):
    error = 0
    mask = val_ratings_mat > 0
    #error = np.sum((val_ratings_mat[mask] - pred_ratings_mat[mask])**2)
    op1 = val_ratings_mat[mask]
    op2 = torch.round(pred_ratings_mat[mask])
    op3 = op1 - op2
    op4 = op3**2
    error = np.sum(op4.detach().cpu().numpy())
    avg_error = error / len(mask.nonzero())
    return avg_error

def loss_fn(train_ratings_mat, pred_ratings_mat, params):
    mask = train_ratings_mat > 0
    rating_loss = torch.norm(pred_ratings_mat[mask] - train_ratings_mat[mask])
    op1 = torch.square(params['ud_vec']).t()  + torch.square(params['md_vec']) # broadcasting
    op2 = torch.unsqueeze(torch.norm(params['ul_mat'], dim=0), 0) 
    op3 = torch.unsqueeze(torch.norm(params['ml_mat'], dim=0), 1) 
    op4 = op1 + op2 + op3 # broadcasting
    reg_loss = torch.sum(op4[mask])
    
    loss = rating_loss + args.regularization * reg_loss
    return loss

def predict_ratings(params):
    op1 = torch.matmul(params['ml_mat'].t(), params['ul_mat'])
    op2 = params['md_vec'] + params['ud_vec'].t() # broadcasting
    pred_ratings_mat = op1 + op2 + params['mu'] # broadcasting
    return pred_ratings_mat

def train_params(train_ratings_mat, val_ratings_mat):
    if args.load_params:
        params = pickle.load(open('params.pkl', 'rb'))
        return params

    dtype = torch.float #FIXME: difference between float and float32?
    device = torch.device('cuda')
    torch.manual_seed(2)
    train_ratings_mat = torch.from_numpy(train_ratings_mat).float().to(device)
    val_ratings_mat = torch.from_numpy(val_ratings_mat).float().to(device)

    torch.autograd.set_detect_anomaly(True)
    params = {
        'mu'     : torch.randn(1, 1, device=device, dtype=dtype, requires_grad=True),
        # user deviation vector
        'ud_vec' : torch.randn(NUM_USERS, 1, device=device, dtype=dtype, requires_grad=True),
        # movie deviation vector
        'md_vec' : torch.randn(NUM_MOVIES, 1, device=device, dtype=dtype, requires_grad=True),
        # user latent matrix
        'ul_mat' : torch.randn(args.latent_vec_size, NUM_USERS, device=device, dtype=dtype, requires_grad=True),
        # movie latent matrix
        'ml_mat' : torch.randn(args.latent_vec_size, NUM_MOVIES, device=device, dtype=dtype, requires_grad=True)}

    optimizer = torch.optim.Adam([params[p] for p in params], lr=args.learning_rate)
    for t in range(args.num_epochs):
        # broadcasting
        pred_ratings_mat = predict_ratings(params)
        loss = loss_fn(train_ratings_mat, pred_ratings_mat, params)
        if t % 50 == 0:
            train_avg_error = avg_error_fn(train_ratings_mat, pred_ratings_mat)
            val_avg_error = avg_error_fn(val_ratings_mat, pred_ratings_mat) 
            print("Epoch {:d}, train loss {:.1f}, train avg_error {:.3f}, val avg_error {:3f}".format(\
                t, loss, train_avg_error, val_avg_error))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with open('params.pkl', 'wb') as f:
        pickle.dump(params, f)

    return params

if __name__ == '__main__':
    all_users = defaultdict(list) # dict: keyed on users, values are lists of the form [(movie_id, rating)...]
    uhash = {}
    hashnum = 0 # each user_id maps to a unique integer between 0 and 5904
    all_ratings = np.zeros([NUM_MOVIES, NUM_USERS])

    with open('../data/train.csv', 'r') as read_obj:
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

    params = train_params(train_ratings_mat, val_ratings_mat)
    pred_ratings_mat = torch.round(predict_ratings(params)).detach().cpu().numpy()
    
    print("Now processing test data.")

    with open('../data/test.csv', 'r') as read_obj:
        with open('output.txt', 'w') as write_obj:
            csv_reader = reader(read_obj)
            for line_num, row in enumerate(csv_reader):
                movie_id, user_id, rating, date = row
                if user_id not in uhash: # this only happens once!
                    user_id = '4597' # some random user
                write_obj.write(str(int(pred_ratings_mat[int(movie_id)-1, uhash[user_id]])) + "\n")
