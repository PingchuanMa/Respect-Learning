from argparse import ArgumentParser
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../'
result_path = base_dir + 'results/'


def plot_rewards(reward_list, title, order=6):
    x = np.arange(len(reward_list))
    plt.figure('Training Result')

    fit = np.polyfit(x, reward_list, order)
    fit = np.polyval(fit, x)

    plt.plot(x, reward_list, color='r', alpha=0.5)
    plt.plot(x, fit, lw=2, label=title, color='r')

    plt.xlabel('Iteration Number')
    plt.ylabel('Episode Reward')
    plt.grid(True)

    save_path = result_path + 'figure/'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + title + ".png")
    plt.close()

def main():

    parser = ArgumentParser(description='Plot.')
    parser.add_argument('--id', type=str, default='origin')
    parser.add_argument('--iter', type=str, default='final')
    parser.add_argument('--order', type=int, default=6)
    args = parser.parse_args()

    reward_json_path = result_path + 'reward/' + args.id + '_' + args.iter + '.json'
    with open(reward_json_path, 'r') as file:
        reward_list = list(json.load(file))
    
    plot_rewards(reward_list, args.id, args.order)


if __name__ == '__main__':
    main()
