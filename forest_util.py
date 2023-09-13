import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import hiive.mdptoolbox as mdptoolbox
# import mdptoolbox
import hiive.mdptoolbox.example
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
import numpy as np
import time
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# Setting random seed
np.random.seed(2021)

# Suppressing warnings issued by pandas
pd.options.mode.chained_assignment = None

colors = {0: 'g', 1:'c'}

labels = {0: 'W', 1: 'C'}

def plot_forest_mgt(policy, title='Forest Management', saveFig=False):
    rows = 25
    cols = 25

    # Reshape policy array to be 2-D
    policy = np.array(list(policy)).reshape(rows, cols)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, xlim=(-0.01, cols+0.01), ylim=(-0.01, rows+0.01))
    plt.title(title, fontsize=16, weight='bold', y=1.01)

    for i in range(25):
        for j in range(25):
            y = 25 - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor='k')
            p.set_facecolor(colors[policy[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, labels[policy[i, j]],
                           horizontalalignment='center', size=10, verticalalignment='center', color='w')
    plt.axis('off')
    if saveFig:
        fig = plt.gcf()
        plt.savefig('figure/forest/' + title + '.png', dpi=400)
        plt.close(fig)
    else:
        plt.show()


##################################
# Policy Iteration
##################################
def policy_iteration(t, r, gammas, max_iterations=10000, display_results=False):
    # Sought inspiration from https://github.com/reedipher
    # /CS-7641-reinforcement_learning/blob/master/code/forest.ipynb

    start_time = time.time()
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct',
               'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)), columns=columns)


    print(f'gamma, \ttime, \titer, \treward')
    print(80*'_')

    test_num = 0
    for g in gammas:
        test = PolicyIteration(t, r, gamma=g, max_iter=max_iterations, eval_type='matrix')

        runs = test.run()
        timing = test.time
        iters = test.iter
        max_r = runs[-1]['Max V']

        # To hold the final results
        max_rewards, mean_rewards, errors = [], [], []
        for run in runs:
            max_rewards.append(run['Max V'])
            mean_rewards.append(run['Mean V'])
            errors.append(run['Error'])

        data['gamma'][test_num] = g
        data['time'][test_num] = timing
        data['iterations'][test_num] = iters
        data['reward'][test_num] = max_r
        data['mean_rewards'][test_num] = {tuple(mean_rewards)}
        data['max_rewards'][test_num] = {tuple(max_rewards)}
        data['error'][test_num] = {tuple(errors)}
        data['policy'][test_num] = {test.policy}

        print('%.2f, \t%.2f,\t%d,\t%f' % (g, timing, iters, max_r))
        policy = data['policy'][test_num]
        title = 'Forest Management'
        if display_results:
            plot_forest_mgt(policy, title, True)
            pass
        test_num += 1

    end_time = time.time() - start_time
    print('Time taken: %.2f' % end_time)

    # Display the differences in policy
    policies = data['policy']

    # Replace all Nan's
    data.fillna(0, inplace=True)
    data.head()

    return data


###################################
# Value Iteration
###################################
def value_iteration(t, r, gammas, epsilons, max_iterations=10000, display_results=False):
    start_time = time.time()

    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct',
               'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas) * len(epsilons)), columns=columns)

    print('Gamma,\tEps,\tTime,\tIter,\tReward')
    print(80 * '_')

    test_num = 0
    for g in gammas:
        for e in epsilons:
            test = ValueIteration(t, r, gamma=g, epsilon=e, max_iter=max_iterations)

            runs = test.run()
            timing = runs[-1]['Time']
            iters = runs[-1]['Iteration']
            max_r = runs[-1]['Max V']

            max_rewards, mean_rewards, errors = [], [], []
            for run in runs:
                max_rewards.append(run['Max V'])
                mean_rewards.append(run['Mean V'])
                errors.append(run['Error'])

            policy = np.array(test.policy)

            data['gamma'][test_num] = g
            data['epsilon'][test_num] = e
            data['time'][test_num] = timing
            data['iterations'][test_num] = iters
            data['reward'][test_num] = max_r
            data['mean_rewards'][test_num] = {tuple(mean_rewards)}
            data['max_rewards'][test_num] = {tuple(max_rewards)}
            data['error'][test_num] = {tuple(errors)}
            data['policy'][test_num] = {test.policy}

            print('%.2f,\t%.0E,\t%.2f,\t%d,\t%f' % (g, e, timing, iters, max_r))
            test_num += 1
    end_time = time.time() - start_time
    print('Time taken: %.2f' % end_time)

    # Display the differences in policy
    policies = data['policy']

    # Replace all NaN's
    data.fillna(0, inplace=True)
    data.head()

    return data



##################
# Q-Learning
###################
def q_learning(t, r, gammas, alphas, alpha_decays=[0.99], epsilon_decays=[0.99], n_iterations=[10000000], display_results=False):

    # Data structure to store hyperparameters
    columns = ['gamma', 'alpha', 'alpha_decay', 'epsilon_decay', 'iterations', 'time', 'reward', 'average_steps',
               'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    size_of_tests = len(gammas) * len(alphas) * len(alpha_decays) * len(epsilon_decays) * len(n_iterations)
    data = pd.DataFrame(0.0, index=np.arange(size_of_tests), columns=columns)

    print('Gamma,\tAlpha, \tTime, \tIter, \tReward')
    print(80*'_')

    test_num = 0
    for g in gammas:
        for a in alphas:
            for a_decay in alpha_decays:
                for e_decay in epsilon_decays:
                    for n in n_iterations:
                        print('Test number %d/%d' % (test_num+1, size_of_tests))
                        print('Gamma: %.2f,\tAlpha: %.2f,\tAlpha Decay:%.3f,\tEpsilon Decay:%.3f,\tIterations:%d'
                              % (g, a, a_decay, e_decay, n))

                        test = QLearning(t, r, gamma=g, alpha=a, alpha_decay=a_decay, epsilon_decay=e_decay, n_iter=n)
                        runs = test.run()
                        timing = runs[-1]['Time']
                        iters = runs[-1]['Iteration']
                        max_r = runs[-1]['Max V']

                        max_rewards, mean_rewards, errors = [], [], []

                        for run in runs:
                            max_rewards.append(run['Max V'])
                            mean_rewards.append(run['Mean V'])
                            errors.append(run['Error'])

                        data['gamma'][test_num] = g
                        data['alpha'][test_num] = a
                        data['alpha_decay'][test_num] = a_decay
                        data['epsilon_decay'][test_num] = e_decay
                        data['time'][test_num] = timing
                        data['iterations'][test_num] = iters
                        data['reward'][test_num] = max_r
                        data['mean_rewards'][test_num] = {tuple(mean_rewards)}
                        data['max_rewards'][test_num] = {tuple(max_rewards)}
                        data['error'][test_num] = {tuple(errors)}
                        data['policy'][test_num] = {test.policy}

                        print('%.2f,\t%.2f,\t%.2f,\t%d,\t%f' % (g, a, timing, iters, max_r))
                        if display_results:
                            pass
                        test_num += 1

    policies = data['policy']


    # Replace all NaN's with 0
    data.fillna(0, inplace=True)
    data.head()

    return data


#######################################
# Plotting of QLearning
########################################
def plot_QL(df, interest, dependent, independent, title=None, logscale=False, saveFig=False):
    if dependent not in interest:
        print(f'Dependent variable is not available')
        return
    if independent not in interest:
        print(f'Independent variable is not available')
        return
    
    x = np.unique(df[dependent])
    y = []

    for i in x:
        y.append(df.loc[df[dependent] == i][independent].mean())

    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, y, 'o-')

    if title == None:
        title = independent + 'vs. ' + dependent
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(dependent)
    plt.ylabel(independent)
    plt.grid(True)
    if logscale:
        plt.xscale('log')

    title = 'QL' + independent + '_vs_' + dependent
    if saveFig:
        fig = plt.gcf()
        fig.savefig('figure/forest/' + title + '.png', dpi=200)
        plt.close()
    else:
        plt.show()






