import itertools

'''
Set up all combination for the hyperparameters
'''
seed = [2023,2024,2025,2026,2027,2028,2029,2030,2031]
expert_episodes = [1]
bc_epochs = [10,20,30,50,100]
bc_batch_size = [64]
bc_hidden_size = [64]
bc_ent_weight = [0.0]
bc_l2_weight = [0.0]
rho = [0.1,0.3,0.5,0.8]

# seed = [0,1,2,3,4,5,6,7,8,9,10,11]
# expert_episodes = [1]
# bc_epochs = [100]
# bc_batch_size = [64]
# bc_hidden_size = [64]
# bc_ent_weight = [0.1]
# bc_l2_weight = [0.01]

# seed = [0,1,2]
# expert_episodes = [1,2]
# rw_epochs = [10,20,50]
# rw_batch_size = [64]
# rw_hidden_size = [64,256]
# rw_lr = [1e-4,1e-5]
# iq_episodes = [100]

# Create the grid
grid = list(itertools.product(seed, expert_episodes, bc_epochs, bc_batch_size, bc_hidden_size, bc_ent_weight, bc_l2_weight, rho))
# grid = list(itertools.product(seed, expert_episodes, rw_epochs, rw_batch_size, rw_hidden_size, rw_lr, iq_episodes))
print('Total number of combinations: {}'.format(len(grid)))
# Save the grid to a file
with open('/home/zli911/imitation/param_files/bc_params_at', 'w') as f:
    for item in grid:
        # save each combination as a line, separated by comma
        f.write("%s\n" % ','.join(map(str, item)))


        