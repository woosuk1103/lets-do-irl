import numpy as np

data = np.load(file="../expert_demo/expert_trajectories.npy") # (10, 20, 401)

for i in range(len(data)):
    for j in range(len(data[0])):
        print(data[i][j])