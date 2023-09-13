from forest_util import *
import numpy as np
import gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from frozen_lake_util import *


def main(saveFig = False):
    env = gym.make('FrozenLake-v1').unwrapped

    env.max_episode_steps = 250

    rows = env.nrow
    cols = env.ncol

    T, R = get_T_R(env)

    plot_lake(env, policy=None, title='Frozen Lake', saveFig=True)

    ######################################
    # Running Policy Iteration
    ######################################
    print(f'Running Policy Iteration on Frozen Lake...')
    # Sought inspiration from https://github.com/reedipher
    # /CS-7641-reinforcement_learning/blob/master/code/frozen.ipynb
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pi_data = policy_iteration(env, T, R, gammas, max_iterations=10000, display_results=True)

    # Plotting
    # Comparing average steps taken vs gamma
    x = gammas
    y = pi_data['average_steps']
    sigma = pi_data['steps_stddev']

    fig = plt.figure(figsize=(6,4))
    plt.plot(x, y, '-o')
    plt.fill_between(x, y - sigma, y + sigma, color='g', alpha=0.1)

    plt.title('Average Steps Taken vs. Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Average Steps')
    plt.grid(True)
    if saveFig:
        name = 'PI_average_steps_vs_gamma'
        fig = plt.gcf()
        fig.savefig('figure/frozen_lake/' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


    # Compare success of runs - the average success percentage for each gamma value
    x1 = pi_data['gamma']
    y1 = pi_data['success_pct']

    fig = plt.figure(figsize=(6,4))
    plt.plot(x1, y1, '-o')
    plt.fill_between(x1, y1 - sigma, y1 + sigma, color='g', alpha=0.1)

    plt.title('Percentage Success vs. Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Success %')
    plt.grid(True)
    if saveFig:
        name = 'PI_success_pct_vs_gamma'
        fig = plt.gcf()
        fig.savefig('figure/frozen_lake/' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


    # Compare iterations of runs
    x2 = pi_data['gamma']
    y2 = pi_data['iterations']

    fig = plt.figure(figsize=(6,4))
    plt.plot(x2, y2, '-o')

    plt.title('Iterations of Runs vs. Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations')
    plt.grid(True)
    plt.ylim([0,10])
    if saveFig:
        name = 'PI_iterations_vs_gamma'
        fig = plt.gcf()
        fig.savefig('figure/frozen_lake/' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()


    # Plotting most successful gamma
    best_run = pi_data['success_pct'].argmax()

    best_policy = pi_data['policy'][best_run]
    # First convert best_policy to a list and then to a np array
    best_policy = np.array(list(best_policy)[0])

    # best_policy now becomes a rows x cols array
    best_policy = best_policy.reshape(rows, cols)

    # Plot the policy
    title = 'Frozen Lake PI Optimal Policy'
    plot_lake(env, best_policy, title, saveFig=True)


    # Writing results to csv file
    PI_results_file = 'output/frozen_lake/PI_results.csv'
    pi_data.to_csv(PI_results_file)

    print('Best Result:\n\tSuccess = %.2f\n\tGamma = %.2f' % (pi_data['success_pct'].max(), pi_data['gamma'][best_run]))

    ######################################
    # Running Value Iteration
    ######################################
    print(f'Running Value Iteration on Frozen Lake...')
    gammas = [0.1, 0.3, 0.6, 0.9]
    epsilons = [1e-2, 1e-5, 1e-8, 1e-12]
    vi_data = value_iteration(env, T, R, gammas, epsilons, max_iterations=10000, display_results=True)


    interest = ['gamma', 'epsilon', 'time', 'iterations', 'reward']
    df = vi_data[interest]
    df.to_csv('output/frozen_lake/VI_convergence_results.csv')

    # Compare the average steps taken vs gammas
    x = gammas
    y = vi_data['average_steps']
    for g in gammas:
        y.append(vi_data.loc[vi_data['gamma'] == g]['average_steps'].mean())

    # sns.set(style="whitegrid")
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(6,4))
    ax = sns.barplot(x, y)
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Average Steps')
    ax.set_title('Average Steps vs Gamma')

    if saveFig:
        name = 'VI_average_steps_vs_gamma'
        fig = plt.gcf()
        fig.savefig('figure/frozen_lake/' + name + '.png', format='png', dpi=120)
        plt.close(fig)
    else:
        plt.show()

    # Average success percentage for each gamma value
    x1 = gammas
    y1 = []
    for g in gammas:
        y1.append(vi_data.loc[vi_data['gamma'] == g]['success_pct'].mean())

    fig = plt.figure(figsize=(6,4))
    ax = sns.barplot(x1, y1)
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Success %')
    ax.set_title('Success Percentage vs. Gamma')

    name = 'VI_success_pct_vs_gamma'
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/frozen_lake/' + name + '.png', format='png', dpi=400)
        plt.close(fig)
    else:
        plt.show()

    # Iterations vs Epsilons
    x2 = vi_data.loc[vi_data['gamma']==0.9]['epsilon']
    y2 = vi_data.loc[vi_data['gamma']==0.9]['iterations']
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x3, y3, '-o')
    plt.title('Iterations vs Epsilons')
    plt.xlabel('Epsilons')
    plt.ylabel('Iterations')
    plt.grid(True)
    plt.xscale('log')
    plt.legend(loc='best')
    name = 'Iterations_vs_epsilons'
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/frozen_lake/' + name + '.png', format='png', dpi=200)
        plt.close(fig)
    else:
        plt.show()


    # Plotting the most successful gamma and epsilon
    best_run = vi_data['success_pct'].argmax()
    best_policy = vi_data['policy'][best_run]
    # First convert best_policy to a list and then to a np array
    best_policy = np.array(list(best_policy)[0])

    # best_policy now becomes a rows x cols array
    best_policy = best_policy.reshape(rows, cols)

    # Plot the policy
    title = 'Frozen Lake VI Optimal Policy'
    plot_lake(env, best_policy, title, saveFig=True)

    # Writing all the results to a csv file
    VI_results_file = 'output/frozen_lake/VI_results.csv'
    vi_data.to_csv(VI_results_file)

    print('Best Result:\n\tSuccess = %.2f\n\tGamma = %.2f\n\tEpsilon= %.E' % (
    vi_data['success_pct'].max(), vi_data['gamma'][best_run], vi_data['epsilon'][best_run]))


    #############################################
    # Running Q-Learning
    #############################################
    print(f'Running Q-learning...')
    gammas = [0.8, 0.9, 0.99]
    alphas = [0.01, 0.1, 0.2]
    alpha_decays = [0.9, 0.999]
    epsilon_decays = [0.9, 0.999]
    iterations = [1e5, 1e6, 1e7]

    ql_data = q_learning(env, T, R, gammas, alphas, alpha_decays=alpha_decays,
                         epsilon_decays=epsilon_decays, n_iterations=iterations, display_results=False)

    # Write all the results to a csv file
    QL_results_file = 'output/frozen_lake/QL_results.csv'
    ql_data.to_csv(QL_results_file)

    interest = ['gamma', 'alpha', 'alpha_decay', 'epsilon_decay', 'iterations', 'reward', 'time', 'success_pct']

    plot_QL(ql_data, interest, 'iterations', 'time', title='Mean Time vs Iterations', logscale=True, saveFig=True)
    plot_QL(ql_data, interest, 'iterations', 'success_pct', title='Mean Success Percentage vs Iterations', logscale=True, saveFig=True)

    # Alpha decay vs success pct
    plot_QL(ql_data, interest, 'alpha_decay', 'success_pct', title='Mean Success Percentage vs Alpha Decay', logscale=True, saveFig=True)

    # Results vs gamma
    plot_QL(ql_data, interest, 'gamma', 'reward', title='Mean Reward vs Gamma', logscale=True, saveFig=True)

    # Plot the most successful gamma
    best_run = ql_data['success_pct'].argmax()

    best_policy = ql_data['policy'][best_run]

    # Re-adjusting since we are reading from a csv file
    # Strip off the first "{" and last "}" as best_policy is presented as a string
    best_policy = best_policy[1:-1]

    # eval changes best_policy to a list
    best_policy = eval(best_policy)

    # Convert this list to an array
    best_policy = np.array(best_policy)

    # best_policy now becomes a rows x cols array to be ready for processing with plot_lake()
    best_policy = best_policy.reshape(rows, cols)

    # Plotting the policy
    title = 'Frozen Lake QL Optimal Policy'
    plot_lake(env, best_policy, title, saveFig=True)

    print('Best Result:\n\tSuccess = %.2f\n\tGamma = %.2f,\n\tAlpha = %.2f,\n\tAlpha Decay: %.3f,\n\tEpsilon Decay: %.3f,\n\tIterations: %.1E'
        % (ql_data['success_pct'].max(), ql_data['gamma'][best_run], ql_data['alpha'][best_run],
           ql_data['alpha_decay'][best_run], ql_data['epsilon_decay'][best_run],ql_data['iterations'][best_run]))

if __name__ == "__main__":
    main(saveFig=True)




































