import os
import warnings
import argparse
import pickle

import pytablewriter
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

parser = argparse.ArgumentParser('Gather results, plot them and create table')
parser.add_argument('-a', '--algos', help='Algorithms to include', nargs='+', type=str)
parser.add_argument('-e', '--env', help='Environments to include', nargs='+', type=str)
parser.add_argument('-f', '--exp_folders', help='Folders to include', nargs='+', type=str)
parser.add_argument('-l', '--labels', help='Label for each folder', nargs='+', type=str)
parser.add_argument('-max', '--max-timesteps', help='Max number of timesteps to display', type=int, default=int(2e6))
parser.add_argument('-min', '--min-timesteps', help='Min number of timesteps to keep a trial', type=int, default=-1)
parser.add_argument('-o', '--output', help='Output filename (pickle file), where to save the post-processed data', type=str)
parser.add_argument('-median', '--median', action='store_true', default=False,
                    help='Display median instead of mean in the table')
parser.add_argument('--no-million', action='store_true', default=False,
                    help='Do not convert x-axis to million')
parser.add_argument('--no-display', action='store_true', default=False,
                    help='Do not show the plots')
args = parser.parse_args()

# Activate seaborn
seaborn.set()
results = {}
post_processed_results = {}

args.algos = [algo.upper() for algo in args.algos]

if args.labels is None:
    args.labels = args.exp_folders

for env in args.env:
    plt.figure(f'Results {env}')
    plt.title(f'{env}BulletEnv-v0', fontsize=14)

    x_label_suffix = '' if args.no_million else '(in Million)'
    plt.xlabel(f'Timesteps {x_label_suffix}', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    results[env] = {}
    post_processed_results[env] = {}

    for algo in args.algos:
        for folder_idx, exp_folder in enumerate(args.exp_folders):

            results[env][f'{args.labels[folder_idx]}-{algo}'] = 0.0
            log_path = os.path.join(exp_folder, algo.lower())

            dirs = [os.path.join(log_path, d) for d in os.listdir(log_path) if env in d]

            max_len = 0
            merged_mean, merged_std = [], []
            merged_timesteps, merged_results = [], []
            last_eval = []
            timesteps = None
            for idx, dir_ in enumerate(dirs):
                try:
                    log = np.load(os.path.join(dir_, 'evaluations.npz'))
                except FileNotFoundError:
                    print("Eval not found for", dir_)
                    continue

                mean_ = np.squeeze(log['results'].mean(axis=1))
                #
                # TODO: Compute standard error for all the runs
                # std_ = np.squeeze(log['results'].std(axis=1)) / np.sqrt(log['results'].shape[1])
                std_ = np.squeeze(log['results'].std(axis=1))

                if mean_.shape == ():
                    continue

                merged_mean.append(mean_)
                merged_std.append(std_)

                max_len = max(max_len, len(mean_))
                if len(log['timesteps']) >= max_len:
                    timesteps = log['timesteps']

                # For post-processing
                merged_timesteps.append(log['timesteps'])
                merged_results.append(log['results'])

                # Truncate the plots
                while timesteps[max_len - 1] > args.max_timesteps:
                    max_len -= 1
                timesteps = timesteps[:max_len]

                if len(log['results']) >= max_len:
                    last_eval.append(log['results'][max_len - 1])
                else:
                    last_eval.append(log['results'][-1])


            # Merge runs with different eval freq:
            # ex: (100,) eval vs (10,)
            # in that case, downsample (100,) to match the (10,) samples
            # Discard all jobs that are < min_timesteps
            min_trials = []
            if args.min_timesteps > 0:
                min_ = np.inf
                for n_timesteps in merged_timesteps:
                    if n_timesteps[-1] >= args.min_timesteps:
                        min_ = min(min_, len(n_timesteps))
                        if len(n_timesteps) == min_:
                            max_len = len(n_timesteps)
                            # Truncate the plots
                            while n_timesteps[max_len - 1] > args.max_timesteps:
                                max_len -= 1
                            timesteps = n_timesteps[:max_len]
                # Downsample if needed
                for trial_idx, n_timesteps in enumerate(merged_timesteps):
                    # We assume they are the same, or they will be discarded in the next step
                    if len(n_timesteps) == min_ or n_timesteps[-1] < args.min_timesteps:
                        pass
                    else:
                        # Discard
                        # merged_mean[trial_idx] = []

                        new_merged_mean, new_merged_std = [], []
                        # Nearest neighbour
                        distance_mat = distance_matrix(n_timesteps.reshape(-1, 1), timesteps.reshape(-1, 1))
                        closest_indices = distance_mat.argmin(axis=0)
                        for closest_idx in closest_indices:
                            new_merged_mean.append(merged_mean[trial_idx][closest_idx])
                            new_merged_std.append(merged_std[trial_idx][closest_idx])
                        merged_mean[trial_idx] = new_merged_mean
                        merged_std[trial_idx] = new_merged_std
                        last_eval[trial_idx] = merged_results[trial_idx][closest_indices[-1]]


            # Remove incomplete runs
            mean_tmp, std_tmp, last_eval_tmp = [], [], []
            for idx in range(len(merged_mean)):
                if len(merged_mean[idx]) >= max_len:
                    mean_tmp.append(merged_mean[idx][:max_len])
                    std_tmp.append(merged_std[idx][:max_len])
                    last_eval_tmp.append(last_eval[idx])
            merged_mean = mean_tmp
            merged_std = std_tmp
            last_eval = last_eval_tmp

            # Post-process
            if len(merged_mean) > 0:
                # shape: (n_trials, n_eval * n_eval_episodes)
                merged_mean = np.array(merged_mean)
                n_trials = len(merged_mean)
                n_eval = len(timesteps)
                # reshape to (n_trials, n_eval, n_eval_episodes)
                evaluations = merged_mean.reshape((n_trials, n_eval, -1))
                # re-arrange to (n_eval, n_trials, n_eval_episodes)
                evaluations = np.swapaxes(evaluations, 0, 1)
                # (n_eval,)
                mean_ = np.mean(evaluations, axis=(1, 2))
                # (n_eval, n_trials)
                mean_per_eval = np.mean(evaluations, axis=-1)
                # (n_eval,)
                std_ = np.std(mean_per_eval, axis=-1)
                # std: error:
                std_error = std_ / np.sqrt(n_trials)
                # Take last evaluation
                # shape: (n_trials, n_eval_episodes) to (n_trials,)
                last_evals = np.array(last_eval).squeeze().mean(axis=-1)
                # Standard deviation of the mean performance for the last eval
                std_last_eval = np.std(last_evals)
                # Compute standard error
                std_error_last_eval = std_last_eval / np.sqrt(n_trials)

                if args.median:
                    results[env][f'{algo}-{args.labels[folder_idx]}'] = f'{np.median(last_evals):.0f}'
                else:
                    results[env][f'{algo}-{args.labels[folder_idx]}'] = f'{np.mean(last_evals):.0f} +/- {std_error_last_eval:.0f}'

                # x axis in Millions of timesteps
                divider = 1e6
                if args.no_million:
                    divider = 1.0

                post_processed_results[env][f'{algo}-{args.labels[folder_idx]}'] = {
                    'timesteps': timesteps,
                    'mean': mean_,
                    'std_error': std_error
                }

                plt.plot(timesteps / divider, mean_, label=f'{algo}-{args.labels[folder_idx]}')
                plt.fill_between(timesteps / divider, mean_ + std_error, mean_ - std_error, alpha=0.5)

    plt.legend()


writer = pytablewriter.MarkdownTableWriter()
writer.table_name = "results_table"

headers = ["Environments"]

# One additional row for the subheader
value_matrix = [[] for i in range(len(args.env) + 1)]

headers = ["Environments"]
# Header and sub-header
value_matrix[0].append('')
for algo in args.algos:
    for label in args.labels:
        value_matrix[0].append(label)
        headers.append(algo)

writer.headers = headers


for i, env in enumerate(args.env, start=1):
    value_matrix[i].append(env)
    for algo in args.algos:
        for label in args.labels:
            key = f'{algo}-{label}'
            value_matrix[i].append(f'{results[env].get(key, "0.0 +/- 0.0")}')

writer.value_matrix = value_matrix
writer.write_table()

post_processed_results['results_table'] = {
    'headers': headers,
    'value_matrix': value_matrix
}

if args.output is not None:
    print(f"Saving to {args.output}.pkl")
    with open(f'{args.output}.pkl', 'wb') as file_handler:
        pickle.dump(post_processed_results, file_handler)

if not args.no_display:
    plt.show()