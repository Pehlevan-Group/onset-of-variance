import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import math
from functools import partial

import jax.numpy as jnp
import jax
from jax.config import config
from jax import random
from jax import jit, grad, vmap, value_and_grad
from jax.tree_util import tree_flatten
from jax.example_libraries import optimizers

import neural_tangents as nt
from neural_tangents import stax


from kernel_generalization.utils import gegenbauer

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

dir_gegenbauer = os.path.join(os.getcwd(),'GegenbauerEigenvalues.npz')

savedir = "/n/holyscratch01/pehlevan_lab/Lab/aatanasov/"

def time_now():
  return datetime.now(pytz.timezone('US/Eastern')).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
  t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
  return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

print('After running code: ',jax.devices())

@jit
def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


# generate P data vectors on the unit sphere in R^D as P x D array
def generate_synth_data(p, dim, key):
  x0 = random.normal(key, shape=(p,dim))
  x = x0 / np.outer(np.linalg.norm(x0, axis=1), np.ones(dim))
  return jnp.array(x)

def pure_target_fn(X, beta, k):
  dim = len(beta)
  z = np.dot(X, beta)
  y = gegenbauer.gegenbauer(z, k+1, dim)[k,:]
  return jnp.array(y)[:, jnp.newaxis]

def generate_train_data(p, beta, k, key):
  dim = len(beta)
  key, emp_key = random.split(key)
  X = generate_synth_data(p, dim, key)
  y = pure_target_fn(X, beta, k)
  return X, y

def format_ps(pvals):
  result = np.zeros(len(pvals), dtype=int)
  for i, p in enumerate(pvals):
    if p < 10:
      result[i] = p + (p % 2)
    elif p < 300:
      result[i] = p + 10 - (p % 10)
    elif p < 3000:
      result[i] = p + 100 - (p % 100)
    else:
      result[i] = p + 1000 - p % 1000
  return result     

# Generate fully connected NN architecture
def fully_connected(num_layers, width, sigma):
  layers = []
  for i in range(num_layers):
    layers += [stax.Dense(width, W_std=sigma, b_std = 0), stax.Relu()]
  layers += [stax.Dense(1, W_std=sigma, b_std=0)] 
  return stax.serial(*layers)

## Kernel Regression Functions
def NTK_expt(kernel_fn, Xs_train, ys_train, X_test, y_test):
  pvals = [X_train.shape[0] for X_train in Xs_train]
  p_test = X_test.shape[0]
  yhats = np.zeros((len(pvals), p_test, 1))
  for i, p in enumerate(pvals):
    X_train = Xs_train[i]
    y_train = ys_train[i]
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, y_train, diag_reg=1e-15)
    yhat = predict_fn(x_test=X_test, get='ntk', compute_cov=False)
    yhats[i, :] = yhat
  gen_error = jnp.array([np.mean((y_test-yhats[i, :])**2) for i in range(len(yhats))])
  return yhats, gen_error
  
def eNTK_expt(apply_fn, params0s, paramsfs, Xs_train, ys_train, X_test, y_test):
  pvals = [X_train.shape[0] for X_train in Xs_train]
  p_test = X_test.shape[0]
  num_repeats = len(params0s[0])
  gen_errs = np.zeros((len(pvals), num_repeats))
  train_errs = np.zeros((len(pvals), num_repeats))
  yhats = np.zeros((len(pvals), num_repeats, p_test, 1))

  for i, p in enumerate(pvals):
    X_train = Xs_train[i]
    y_train = ys_train[i]
    for j in range(num_repeats):
      params0 = params0s[i][j]
      paramsf = paramsfs[i][j]
      t_start = datetime.now()
      used_fn = lambda params, X: apply_fn(params, X) - apply_fn(params0, X)
      kernel_fn = nt.batch(nt.empirical_kernel_fn(used_fn, vmap_axes=0, implementation=3), store_on_device=False)
  
      ntk_train_train = kernel_fn(X_train, None, 'ntk', paramsf)
      ntk_test_train = kernel_fn(X_test, X_train, 'ntk', paramsf)
      mse_predictor = nt.predict.gradient_descent_mse(ntk_train_train, y_train)
      
      t = None
      y_train_0 = used_fn(paramsf, X_train)
      y_test_0 = used_fn(paramsf, X_test)
      y_train_f, yhat = mse_predictor(t, y_train_0, y_test_0, ntk_test_train)
      yhats[i,j,:] = yhat
      gen_errs[i,j] = np.mean((y_test-yhat)**2)
      train_errs[i,j] = np.mean((y_train_f-y_train)**2)
      t_end = datetime.now()
      
      string_print = "\r P = %d, gen error: %0.3e | train error: %0.3e | Repeat: %d/%d | Time Elapsed: " % (p, gen_errs[i,j], train_errs[i,j], j+1, num_repeats)
      sys.stdout.write(string_print + time_diff(t_start,t_end) + "\n")
      
  return {"yhats_eNTK": yhats, "tr_err_eNTK": train_errs, "gen_err_eNTK": gen_errs}


# Neural Network Training Functions
@partial(jit, static_argnums=(1,2,3,6,7))
def train_nn(init_key, init_fn, apply_fn, optimizer, X_test, train_set, 
             num_iter=30000, min_loss=1e-6):
    """
    Training the neural network, possibly as a linear model
    """ 
    _, params0 = init_fn(init_key, (-1, train_set[0].shape[1]))

    opt_init, opt_update, get_params = optimizer
    opt_state = opt_init(params0)

    used_fn = jit(lambda params, X: apply_fn(params, X) - subtract * apply_fn(params0, X))
    nn_loss = jit(lambda params, X, y: jnp.mean((used_fn(params, X) - y)**2))
    value_and_grad_loss = jit(lambda state, X, y: value_and_grad(nn_loss)(get_params(state), X, y))

    losses = jnp.zeros(num_iter)
    losses = losses.at[0].set(1.0)
    
    init_state = (0, losses, opt_state)
    
    @jit 
    def cond_fn(state):
      t = state[0]
      return (t < num_iter) & (state[1][t] >= min_loss) & (~jnp.isnan(state[1][t]))
    
    @jit
    def iter_fn(state):
      t = state[0]
      loss, grad = value_and_grad_loss(state[2], *train_set)
      return t+1, state[1].at[t+1].set(loss), opt_update(t, grad, state[2])
      
    _, losses, opt_state = jax.lax.while_loop(cond_fn, iter_fn, init_state)
    
    params_f = get_params(opt_state)
    yhat = used_fn(params_f, X_test)
    return params0, params_f, losses, yhat


def nn_expt(architecture, init_keys, Xs_train, ys_train, X_test, y_test, lr=1.0, optim="sgd",
            num_iter=10000, min_loss=1e-5, verbose=True, params_is=None):
  """
  Experiment for sample wise generalization error 
  for overparameterized model
  """
  pvals = [X_train.shape[0] for X_train in Xs_train]
  p_test = X_test.shape[0]
  num_repeats = init_keys.shape[0]

  gen_errs = np.zeros((len(pvals), num_repeats))
  yhats = np.zeros((len(pvals), num_repeats, p_test, 1))
  params_0s, params_fs, train_errs = [], [], []

  if optim == "adam": 
    optimizer = optimizers.adam(lr)
  else: 
    optimizer = optimizers.sgd(lr)

  depth, width, alpha = architecture
  init_fn, apply_fn, _ = fully_connected(depth-1, width, jnp.power(alpha, 1/depth))

  beginning = datetime.now()
  train_fn = jit(vmap(train_nn, (0, None, None, None, None, None)), static_argnums=(1,2,3))

  for i, p in enumerate(pvals):
    t_start = datetime.now()
    train_set = Xs_train[i], ys_train[i]
    output = train_fn(init_keys, init_fn, apply_fn, optimizer, X_test, train_set)
    params_0, params_f, train_losses, yhat = output
    yhats[i] = yhat
    gen_errs[i] = jnp.mean((yhat - y_test)**2, axis=(-1, -2))
    params_0s += [tree_unstack(params_0)]
    params_fs += [tree_unstack(params_f)]
    train_errs += [train_losses]
    t_end = datetime.now()
    
    idxs = jnp.argmax(train_losses==0, axis=1)-1
    idxs = jax.lax.map(lambda x: jnp.where(x<0, num_iter-1, x), idxs)
    tr_err = jnp.mean(train_losses[np.arange(num_repeats), idxs])
    gen_err = jnp.mean(gen_errs[i])

    sys.stdout.write(f" P = {p}, gen_err: {gen_err:.3e} | tr_err: {tr_err:.3e} | T = {max(idxs)} | lr: {lr:.3e} | time elapsed: {time_diff(t_start,t_end)} \n" )
    if verbose:
      idx = max(idxs)
      itr = np.arange(1, idx+1)
      plt.semilogy(itr, train_losses[:, :idx].T,'o')
      plt.show()

    t_end = datetime.now()

  ending = datetime.now()
  print(' | Total Time Elapsed: ' + time_diff(beginning,ending))

  return {"params_0": params_0s, "params_f": params_fs, "yhats_NN": yhats, "tr_err_NN": train_errs, "gen_err_NN": gen_errs}

if __name__ == "__main__":
  ## Parse the arguments
  parser = argparse.ArgumentParser(description='Feed in k and alpha')
  parser.add_argument('-L', type=int, required=True, help="Depth")
  parser.add_argument('-D', type=int, required=True, help="Input Dimension")
  parser.add_argument('-N', type=int, required=False, default=1000, help="Width")
  parser.add_argument('-k', type=int, required=True, help="Task")
  parser.add_argument('-s', type=float, required=True, help="Initialization scale")
  parser.add_argument('-n', type=int, required=False, default=3, help="Num repeats of NN")
  parser.add_argument('-i', type=int, required=False, default=0, help="Init key")
  parser.add_argument('-d', type=int, required=False, default=0, help="Data key")
  parser.add_argument('--subtract', type=int, required=False, default=1, help="Subtract f0")
  args = parser.parse_args()

  L, N, k, alpha, dim = args.L, args.N, args.k, args.s, args.D
  num_repeats, i_key, d_key, subtract = args.n, args.i, args.d, args.subtract
  print(f"D = {dim}, N = {N}, L = {L}, k = {k}, alpha = {alpha}, num_repeats = {num_repeats}, d = {d_key}, subtract = {subtract}")


  ## Experiment Parameters
  num_iter_nn = int(3e4)
  min_loss_nn = 1e-6

  ## NN Hyperparameters
  layers = L - 1   # Number of hidden layers
  depth = L

  ## Dimension, sample sizes, max eigenvalue mode to generate data
  # dim = 10
  num_p = 15
  num_n = 5
  logpmin = .5
  lognmin = 1.5
  logpmax = np.log10(10000-1)
  lognmax = np.log10(1000)
  p_test = 2000

  ensemble_size = 1

  # This is the sweep that we are going to be doing:
  pvals = np.logspace(logpmin, logpmax, num=num_p).astype('int')
  nvals = (np.logspace(lognmin, lognmax, num=num_n).astype('int'))[-1:]
  pvals = format_ps(pvals)

  ## Target function mode and label noise
  noise_num = 1
  shift = np.array([-1]) 

  # Random keys
  data_key, init_key = random.PRNGKey(d_key), random.PRNGKey(i_key)
  init_keys = random.split(init_key, num_repeats)
  train_key = random.split(data_key, len(pvals))

  # Crucially independent test key!! 
  test_key = random.PRNGKey(0)
  beta_key, const_key, test_key = random.split(test_key, 3)

  # Data:
  beta = generate_synth_data(1, dim, beta_key)[0,:]
  y_const = jnp.sqrt(jnp.mean((generate_train_data(1000, beta, k, const_key)[1])**2))
  Xs_train = []; ys_train = []
  for i, p in enumerate(pvals):
    X_train, y_train = generate_train_data(p, beta, k, train_key[i])
    y_train = y_train/y_const
    Xs_train += [X_train]
    ys_train += [y_train]
  X_test, y_test = generate_train_data(p_test, beta, k, test_key)
  y_test = y_test/y_const
  test_set = X_test, y_test

  # Infinite width NTK Regression: 
  print('NTK Regression Start @ {}'.format(time_now()))
  width = 10
  init_fn, apply_fn, kernel_fn = fully_connected(layers, width, jnp.power(alpha, 1/depth))
  apply_fn = jit(apply_fn)
  kernel_fn = jit(kernel_fn, static_argnums=(2,))
  t_start = datetime.now()
  yhats, err_regression = NTK_expt(kernel_fn, Xs_train, ys_train, X_test, y_test)
  jnp.save(savedir+f"inf_err_D={dim}_L={depth}_k={k}_s={alpha:.2f}_d={d_key}", err_regression)
  jnp.save(savedir+f"pred_inf_D={dim}_L={depth}_k={k}_s={alpha:.2f}_d={d_key}", yhats)
  print('Time Elapsed: ' + time_diff(t_start, datetime.now()))

  subtract_str = "" if subtract else "_unsub"
  nvals = [N]
  for width in nvals:
    n_start_time = datetime.now()

  # create NN and kernel function
    init_fn, apply_fn, kernel_fn = fully_connected(layers, width, jnp.power(alpha, 1/depth))
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    # NN experiment:
    t_start = datetime.now()
    print(f'N = {width}, steps = {num_iter_nn} Neural Network Training Start @ {t_start}')
    # Set learning rate: 
    lr = jnp.sqrt(width) / ((0.3)**2 + max(L-3, 0)*10/width +  alpha**(2))
    architecture = depth, width, alpha
    data = nn_expt(architecture, init_keys, Xs_train, ys_train, X_test, y_test, lr=lr, optim="sgd",
              num_iter=num_iter_nn, min_loss=min_loss_nn, verbose=False)
    for key in data:
      jnp.save(savedir+key+f"_D={dim}_N={width}_L={depth}_k={k}_s={alpha:.2f}_d={d_key}"+subtract_str, data[key])
    params_0s, params_fs = data['params_0'], data['params_f']
    print('NN Time Elapsed: ' + time_diff(t_start, datetime.now()))

    # eNTK Experiments: 
    t_start = datetime.now()
    print(f'N = {width} eNTK0 Training Start @ {t_start}')
    data = eNTK_expt(apply_fn, params_0s, params_0s, Xs_train, ys_train, X_test, y_test)
    yhats_eNTK0, tr_errs_eNTK0, gen_errs_eNTK0 = data
    for key in data:
      jnp.save(savedir+key+f"0_D={dim}_N={width}_L={depth}_k={k}_s={alpha:.2f}_d={d_key}", data[key])
    print('eNTK0 Time Elapsed: ' + time_diff(t_start, datetime.now()))
    
    t_start = datetime.now()
    print('N = {} eNTKf Training Start @ {}'.format(width, t_start))
    data = eNTK_expt(apply_fn, params_0s, params_fs, Xs_train, ys_train, X_test, y_test)
    yhats_eNTKf, tr_errs_eNTKf, gen_errs_eNTKf = data
    for key in data:
      jnp.save(savedir+key+f"f_D={dim}_N={width}_L={depth}_k={k}_s={alpha:.2f}_d={d_key}", data[key])
    print('eNTKf Time Elapsed: ' + time_diff(t_start, datetime.now()))

    print("Total Time elapsed: " + time_diff(n_start_time, datetime.now()))

    sys.stdout.flush()



