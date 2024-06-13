import numpy as np
import os
import pickle

def process_results(result_path):
    '''
    Process result files under result_path
    Summarize all files in one and find the best model
    '''
    # Get all files
    file_list = os.listdir(result_path)
    print('Total number of files: {}'.format(len(file_list)))
    # Read all files and append to a list
    # files are saved as npz files
    # Each file contains two arrays: returns and timesteps
    # files names are like: bc_result_50_20_32_64_0.001_0.01_seed_0.npz
    # Group files with the same hyperparameters but different seeds
    # Compute the mean and std of returns and timesteps for each group
    file_dict = {}
    episodes_list = []
    epochs_list = []
    for file in file_list:
        if 'AT' not in file:
            continue
        if '.npz' not in file:
            continue
        # if '64_64_0.01_0.01' not in file:
        #     continue
        # group files with the same hyperparameters but different seeds
        # hyperparameters are: expert_episodes, bc_epochs, bc_batch_size, bc_hidden_size, bc_ent_weight, bc_l2_weight
        params = file[:-5].split('_')
        params = params[2:9]
        params = tuple(params)
        if params[-1] != '0.1':
            continue
        if params not in file_dict.keys():
            file_dict[params] = [file]
            episodes_list.append(int(params[0]))
            epochs_list.append(int(params[1]))
        else:
            file_dict[params].append(file)
    print(file_dict)
    episodes_list = np.unique(episodes_list)
    epochs_list = np.unique(epochs_list)

    returns_list = []
    timesteps_list = []
    mean_returns_list = []
    std_returns_list = []
    mean_timesteps_list = []
    std_timesteps_list = []
    params_list = []

    bc_returns_list = []
    bc_timesteps_list = []
    iq_returns_list = []
    iq_timesteps_list = []

    episodes_index_list = [[] for i in range(len(episodes_list))]
    epochs_index_list = [[] for i in range(len(epochs_list))]
    for i, params in enumerate(file_dict.keys()):
        returns_list.append([])
        timesteps_list.append([])
        bc_returns_list.append([])
        bc_timesteps_list.append([])
        iq_returns_list.append([])
        iq_timesteps_list.append([])
        params_list.append(params)
        # Read all files with the same hyperparameters
        for file in file_dict[params]:
            file_path = os.path.join(result_path, file)
            data = np.load(file_path)
            returns_list[i].append(data['returns'])
            timesteps_list[i].append(data['timesteps'])
            if 'bc_returns' in data.keys():
                bc_returns_list[i].append(data['bc_returns'])
                bc_timesteps_list[i].append(data['bc_timesteps'])
            if 'iq_returns' in data.keys():
                iq_returns_list[i].append(data['iq_returns'])
                iq_timesteps_list[i].append(data['iq_timesteps'])
        mean_returns_list.append(np.mean(returns_list[i]))
        std_returns_list.append(np.std(returns_list[i]))
        mean_timesteps_list.append(np.mean(timesteps_list[i]))
        std_timesteps_list.append(np.std(timesteps_list[i]))
        # Find the index of the corresponding expert_episodes
        episodes_index_list[np.where(episodes_list == int(params[0]))[0][0]].append(i)
        epochs_index_list[np.where(epochs_list == int(params[1]))[0][0]].append(i)

    mean_returns_list = np.array(mean_returns_list)
    std_returns_list = np.array(std_returns_list)
    mean_timesteps_list = np.array(mean_timesteps_list)
    std_timesteps_list = np.array(std_timesteps_list)

    # Find the best model for each expert episodes num and print the results with best hyperparameters
    for i, index_list in enumerate(epochs_index_list):
        # print(index_list)
        best_index = np.argmax(mean_returns_list[index_list])
        # print('Expert Episodes: {}'.format(episodes_list[i]))
        print('Epochs: {}'.format(epochs_list[i]))
        print('Best model: {}'.format(file_dict[params_list[index_list[best_index]]]))
        print('Mean returns: {} +/- {}'.format(mean_returns_list[index_list[best_index]], std_returns_list[index_list[best_index]]))
        print('Mean timesteps: {} +/- {}'.format(mean_timesteps_list[index_list[best_index]], std_timesteps_list[index_list[best_index]]))
        # print(np.array(returns_list[index_list[best_index]]))
        print(np.mean(returns_list[index_list[best_index]],1,keepdims=False))
        if len(bc_returns_list[0]) > 0:
            print('Mean BC returns: {} +/- {}'.format(np.mean(bc_returns_list[index_list[best_index]]), np.std(bc_returns_list[index_list[best_index]])))
            print('Mean BC timesteps: {} +/- {}'.format(np.mean(bc_timesteps_list[index_list[best_index]]), np.std(bc_timesteps_list[index_list[best_index]])))
            print(np.mean(bc_returns_list[index_list[best_index]],1,keepdims=False))
        if len(iq_returns_list[0]) > 0:
            print('Mean IQ returns: {} +/- {}'.format(np.mean(iq_returns_list[index_list[best_index]]), np.std(iq_returns_list[index_list[best_index]])))
            print('Mean IQ timesteps: {} +/- {}'.format(np.mean(iq_timesteps_list[index_list[best_index]]), np.std(iq_timesteps_list[index_list[best_index]])))
            print(np.mean(iq_returns_list[index_list[best_index]],1,keepdims=False))
    # Save the results
    with open(os.path.join(result_path, 'summary.pkl'), 'wb') as f:
        pickle.dump({'file_prefix_list': params_list, 'mean_returns_list': mean_returns_list, 'std_returns_list': std_returns_list, 'mean_timesteps_list': mean_timesteps_list, 'std_timesteps_list': std_timesteps_list}, f)
    print('Results saved to {}'.format(os.path.join(result_path, 'summary.pkl')))

if __name__ == "__main__":
    result_path = '/home/zli911/imitation/result_files/PongNoFrameskip-v4/bc_ppo'
    # result_path = '/home/zli911/imitation/result_files/HopperBulletEnv-v0/bc_sac'
    process_results(result_path)