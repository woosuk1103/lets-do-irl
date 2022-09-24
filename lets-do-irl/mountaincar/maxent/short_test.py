import numpy as np

state = []
for i in range(3):
    state.append([0])

state[0]  = [1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0]
state[1]  = [0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
state[2]  = [1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0]

state = np.array(state)
print(type(state))

print(np.shape(state))

state = np.reshape(state, (-1,1))
print(np.shape(state))

print(state[3]==1)