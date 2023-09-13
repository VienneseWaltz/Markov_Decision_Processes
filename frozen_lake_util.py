import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
import time



def plot_lake(env, policy=None, title='Frozen Lake', saveFig=False):
    # Credit goes to https://github.com/reedipher
    # /CS-7641-reinforcement_learning/blob/master/code/frozen.ipynb
    colors = {
        b'S': 'b',
        b'F': 'w',
        b'H': 'k',
        b'G': 'g'
    }

    directions = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑'
    }
    squares = env.nrow
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, xlim=(-.01, squares+0.01), ylim=(-.01, squares+0.01))
    plt.title(title, fontsize=16, weight="bold", y=1.01)
    for i in range(squares):
        for j in range(squares):
            y = squares - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor='k')
            p.set_facecolor(colors[env.desc[i, j]])
            ax.add_patch(p)

            if policy is not None:
                text = ax.text(x + 0.5, y + 0.5, directions[policy[i, j]],
                               horizontalalignment='center', size=25, verticalalignment='center',
                               color='k')
    plt.axis('off')
    if saveFig:
        fig = plt.gcf()
        plt.savefig('figure/frozen_lake/' + title + '.png', dpi=400)
        plt.close(fig)
    else:
        plt.show()


def get_score(env, policy, episodes=1000, print_info=True):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:
            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                print(f'Yay! You have retrieved the Frisbee after {steps} steps')
                steps_list.append(steps)
                break
            elif done and reward == 0:
                print(f'You fell into a hole.')
                misses += 1
                break

    avg_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    pct_failure = (misses/episodes) * 100

    if print_info:
        print('**********************************************')
        print('You took an average of {:.0f} steps to get the frisbee'.format(np.mean(steps_list)))
        print('And you fell in the hole {:.2f} % of the times'.format((misses / episodes) * 100))
        print('***********************************************')

    print(f'Average steps = {avg_steps}')
    print(f'Std steps = {std_steps}')
    print(f'Pct Failure = {pct_failure}')
    return avg_steps, std_steps, pct_failure


def get_T_R(env):
    T = np.zeros((4, env.observation_space.n, env.observation_space.n))
    R = np.zeros((4, env.observation_space.n, env.observation_space.n))

    for state in env.P:
        for action in env.P[state]:
            for (prob, next_state, reward, _) in env.P[state][action]:
                T[action][state][next_state] += prob
                R[action][state][next_state] += reward
    return T, R



##################################
# Policy Iteration
##################################
def policy_iteration(env, t, r, gammas, max_iterations=10000, display_results=False):
    start_time = time.time()
    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct',
               'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas)), columns=columns)

    print(f'gamma, \ttime, \titer, \treward')
    print(80 * '_')

    rows = 4
    cols = 4
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

        policy = np.array(test.policy)
        policy = policy.reshape(4,4)

        data['gamma'][test_num] = g
        data['time'][test_num] = timing
        data['iterations'][test_num] = iters
        data['reward'][test_num] = max_r
        data['mean_rewards'][test_num] = {tuple(mean_rewards)}
        data['max_rewards'][test_num] = {tuple(max_rewards)}
        data['error'][test_num] = {tuple(errors)}
        data['policy'][test_num] = {test.policy}

        print('%.2f, \t%.2f,\t%d,\t%f' % (g, timing, iters, max_r))
        # policy = data['policy'][test_num]
        title = 'Frozen_Lake_Policy_Iteration'+ str(rows) + 'x' + str(cols) + '_g' + str(g)
        if display_results:
            plot_lake(env, policy, title, saveFig=True)
            pass
        test_num += 1

    end_time = time.time() - start_time
    print('Time taken: %.2f' % end_time)

    # Display the differences in policy
    policies = data['policy']

    for i, p in enumerate(policies):
        pol = list(p)[0]
        steps, steps_stddev, failures = get_score(env, pol, 1000, print_info=True)
        data['average_steps'][i] = steps
        data['steps_stddev'][i] = steps_stddev
        data['success_pct'][i] = 100 - failures

    # Replace all Nan's
    data.fillna(0, inplace=True)
    data.head()

    return data


###################################
# Value Iteration
###################################
def value_iteration(env, t, r, gammas, epsilons, max_iterations=10000, display_results=False):
    start_time = time.time()

    columns = ['gamma', 'epsilon', 'time', 'iterations', 'reward', 'average_steps', 'steps_stddev', 'success_pct',
               'policy', 'mean_rewards', 'max_rewards', 'error']
    data = pd.DataFrame(0.0, index=np.arange(len(gammas) * len(epsilons)), columns=columns)

    print('Gamma,\tEps,\tTime,\tIter,\tReward')
    print(80 * '_')

    rows = 4
    cols = 4
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
            policy = policy.reshape(4,4)

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
            if display_results:
                title = 'Frozen_Lake_VI_' + str(rows) + 'x' + str(cols) + '_g' + str(g) + '_e' + str(e)
                plot_lake(env, policy, title, saveFig=True)

            test_num += 1
    end_time = time.time() - start_time
    print('Time taken: %.2f' % end_time)

    # Display the differences in policy
    policies = data['policy']

    for i, p in enumerate(policies):
        pol = list(p)[0]
        steps, steps_stddev, failures = get_score(env, pol, 1000, print_info=True)
        data['average_steps'][i] = steps
        data['steps_stddev'][i] = steps_stddev
        data['success_pct'][i] = 100 - failures

    # Replace all NaN's
    data.fillna(0, inplace=True)
    data.head()

    return data


##################
# Q-Learning
###################
def q_learning(env, t, r, gammas, alphas, alpha_decays=[0.99], epsilon_decays=[0.99], n_iterations=[10000000], display_results=False):
    # Data structure to store hyperparameters
    columns = ['gamma', 'alpha', 'alpha_decay', 'epsilon_decay', 'iterations', 'time', 'reward', 'average_steps',
               'steps_stddev', 'success_pct', 'policy', 'mean_rewards', 'max_rewards', 'error']
    size_of_tests = len(gammas) * len(alphas) * len(alpha_decays) * len(epsilon_decays) * len(n_iterations)
    data = pd.DataFrame(0.0, index=np.arange(size_of_tests), columns=columns)

    print('Gamma,\tAlpha, \tTime, \tIter, \tReward')
    print(80*'_')

    rows = 4
    cols = 4

    test_num = 0
    for g in gammas:
        for a in alphas:
            for a_decay in alpha_decays:
                for e_decay in epsilon_decays:
                    for n in n_iterations:
                        print('Test number %d/%d' % (test_num + 1, size_of_tests))
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

                        policy = np.array(test.policy)
                        policy = policy.reshape(rows, cols)

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
                            title = 'Frozen_Lake_QL_' + str(rows) + 'x' + str(cols) + '_g' + str(g) + '_a' + str(a) + '_adecay' + str(a_decay)
                            plot_lake(env, policy=policy, title=title,  saveFig=True)
                        test_num += 1

    policies = data['policy']

    for i, p in enumerate(policies):
        pol = list(p)[0]
        steps, steps_stddev, failures = get_score(env, pol, episodes=1000, print_info=True)
        data['average_steps'][i] = steps
        data['steps_stddev'][i] = steps_stddev
        data['success_pct'][i] = 100 - failures

    # Replace all NaN's
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
        fig.savefig('figure/frozen_lake/' + title + '.png', dpi=200)
        plt.close()
    else:
        plt.show()




































