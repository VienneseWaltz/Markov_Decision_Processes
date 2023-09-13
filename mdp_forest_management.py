import hiive.mdptoolbox.example
from forest_util import *
import numpy as np
import pandas as pd


def main(saveFig = False):
    # Set up the forest management problem
    # Transition probability P and reward matrix R
    P, R = hiive.mdptoolbox.example.forest(S=625)

    # Sought inspiration from https://github.com/reedipher
    # /CS-7641-reinforcement_learning/blob/master/code/forest.ipynb
    # Running Policy Iteration
    print(f'Running Policy Iteration on Forest Management...')
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    pi_data = policy_iteration(P, R, gammas, 10000, display_results=False)
    pi_data.plot(x='gamma', y='reward', title='Rewards vs. Gamma')
    plt.grid()
    plt.show()
    pi_data.plot(x='iterations', y='reward', title='Rewards vs. Iterations')
    plt.grid()
    plt.show()

    # Obtaining best run and best policy for Policy Iteration
    best_run = pi_data['reward'].argmax()
    best_policy = pi_data['policy'][best_run]
    title = 'Forest Management - Optimal Policy for Policy Iteration'
    plot_forest_mgt(best_policy, title, True)

    # Printing best_policy and best_run
    print('Best Result:\n\tReward = %.2f\n\tGamma = %.3f' % (pi_data['reward'].max(), pi_data['gamma'][best_run]))


    # Running Value Iteration
    print(f'Running Value Iteration on Forest Management...')
    gammas = [0.1, 0.3, 0.6, 0.9, 0.9999999]
    epsilons = [1e-2, 1e-3, 1e-8, 1e-12]
    vi_data = value_iteration(P, R, gammas, epsilons, 10000, display_results=False)
    policies = vi_data['policy']
    vi_data.plot(x='gamma', y='reward', title='Reward vs Gammas')
    plt.grid()
    plt.show()
    vi_data.plot(x='iterations', y='reward', title='Reward vs Iterations')
    plt.grid(True)
    plt.show()

    # Obtaining best run and best policy for Value Iteration
    best_run = vi_data['reward'].argmax()
    best_policy = vi_data['policy'][best_run]
    title = 'Forest Management - Optimal Policy for Value Iteration'
    plot_forest_mgt(best_policy, title, True)

    # Save the Value Iteration results to a csv file
    #vi_data.to_csv('forest/Value_Iteration_results.csv')
    print('Result:\n\tBest Reward = %.2f\n\tBest Gamma = %.7f\n\tBest Epsilon= %.E' % (vi_data['reward'].max(),
                                                                             vi_data['gamma'][best_run],
                                                                             vi_data['epsilon'][best_run]))


    # Running Q-Learning
    print(f'Running Q-Learning on Forest Management...')
    gammas = [0.8, 0.9, 0.99]
    alphas = [0.01, 0.1, 0.2]
    alpha_decays = [0.9, 0.999]
    epsilon_decays = [0.9, 0.999]
    iterations = [1e5, 1e6, 1e7]
    ql_data = q_learning(P, R, gammas, alphas, alpha_decays=alpha_decays, epsilon_decays=epsilon_decays,
                         n_iterations=iterations, display_results=False)

    # Save the Q-Learning results to a csv file
    ql_data.to_csv('output/forest/ql_data.csv')


    interest = ['gamma', 'alpha', 'alpha_decay', 'epsilon_decay', 'iterations', 'reward', 'time', 'success_pct']
    plot_QL(ql_data, interest, 'iterations', 'time', title='Mean Time vs Iterations', logscale=True, saveFig=True)
    plot_QL(ql_data, interest, 'iterations', 'reward', title='Mean Reward vs Iterations', logscale=True, saveFig=True)

    # Alpha decay vs reward
    plot_QL(ql_data, interest, 'alpha_decay', 'reward', title='Mean Reward vs Alpha Decay', saveFig=True)

    # Reward vs gamma
    plot_QL(ql_data, interest, 'gamma', 'reward', title='Mean Reward vs Gamma', saveFig=True)

    # Plot the most successful gamma
    best_run = ql_data['reward'].argmax()

    best_policy = ql_data['policy'][best_run]

    # Re-adjusting since we are reading from a csv file
    # Strip off the first "{" and last "}" as best_policy is presented as a string
    best_policy = best_policy[1:-1]

    # eval changes best_policy to a list
    best_policy = eval(best_policy)

    # Convert this list to an array
    best_policy = np.array(best_policy)

    # Plotting the policy
    title = 'Forest Management QL Optimal Policy'
    plot_forest_mgt(best_policy, title, saveFig=True)
 
    print('Best Result:\n\tReward = %.2f\n\tGamma = %.2f,\n\tAlpha = %.2f,\n\tAlpha Decay: %.3f,\n\tEpsilon Decay: %.3f,\n\tIterations: %.1E'
        % (ql_data['reward'].max(), ql_data['gamma'][best_run], ql_data['alpha'][best_run], ql_data['alpha_decay'][best_run],
        ql_data['epsilon_decay'][best_run], ql_data['iterations'][best_run]))


if __name__ == "__main__":
    main(saveFig=True)











































