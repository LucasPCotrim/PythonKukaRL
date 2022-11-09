import matplotlib.pyplot as plt
import numpy as np

def print_eval_results(eval_filepath):
    plt.style.use('fivethirtyeight')

    # Load evaluation.npz
    evaluations = np.load(eval_filepath)
    for k in evaluations.files:
      print(k)
    print(evaluations['ep_lengths'])
    # Obtain timesteps
    timesteps = evaluations['timesteps'].reshape(-1,1)
    # Obtain Rewards
    rewards = evaluations['results']
    rewards = np.mean(rewards, axis=1)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(timesteps, rewards)
    fig.suptitle('Avg. Evaluation rewards')
    ax.set_xlabel('timestep t')
    ax.set_ylabel('rewards')
    plt.show()
    
    return rewards