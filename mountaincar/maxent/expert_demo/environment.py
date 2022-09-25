import math
from typing import Optional

import numpy as np

import plotly.express as px
import cufflinks as cf
cf.go_offline(connected=True)

import gym
import copy
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class Moduleviser(gym.Env):
    """
    ### Description
    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with discrete actions.
    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```
    ### Observation Space
    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
    | 1   | velocity of the car                  | -Inf | Inf | position (m) |
    ### Action Space
    There are 3 discrete deterministic actions:
    | Num | Observation             | Value | Unit         |
    |-----|-------------------------|-------|--------------|
    | 0   | Accelerate to the left  | Inf   | position (m) |
    | 1   | Don't accelerate        | Inf   | position (m) |
    | 2   | Accelerate to the right | Inf   | position (m) |
    ### Transition Dynamics:
    Given an action, the mountain car follows the following transition dynamics:
    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*
    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
    upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
    velocity is clipped to the range `[-0.07, 0.07]`.
    ### Reward:
    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
    penalised with a reward of -1 for each timestep.
    ### Starting State
    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
    The starting velocity of the car is always assigned to 0.
    ### Episode End
    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 200.
    ### Arguments
    ```
    gym.make('MountainCar-v0')
    ```
    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        # self.min_position = -1.2
        # self.max_position = 0.6
        # self.max_speed = 0.07
        # self.goal_position = 0.5
        # self.goal_velocity = goal_velocity

        # self.force = 0.001
        # self.gravity = 0.0025

        self.CE = 0

        self.low = np.zeros(shape=(400,), dtype=np.int32)
        self.high = np.ones(shape=(400,), dtype=np.int32)

        # self.render_mode = render_mode

        # self.screen_width = 600
        # self.screen_height = 400
        # self.screen = None
        # self.clock = None
        # self.isopen = True

        self.action_space = spaces.Discrete(399)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
      
        state = np.reshape(self.state, (20,20))

        row, col = divmod(action,20)
        # modify the state (20 by 20 matrix)   
        # rule-based method
        
        if state[row][col] == 0:
            state[row][col] == 1
        else:
            state[row][col] == 0


        self.state, self.CE = self.clustering(state)###############################

        fig = px.imshow(self.state, text_auto=True, labels=dict(x="components", y="components", color="I/F"),
                x=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                   '18', '19'],
                y=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                   '18', '19'])

        # fig.update_layout(height=800, title_text=self.CE)
        # fig.update_xaxes(side="top")
        fig.show()

        for i in range(len(self.state)):
            print(self.state[i])
        print("CE:",self.CE)
        print("=====Clustering over=====")
        terminated = bool(self.CE >= 0.08)

        reward = -1.0

        # self.state = (position, velocity)
        # if self.render_mode == "human":
        #     self.render()
        return np.array(self.state, dtype=np.float32), self.CE , reward, terminated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        # self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        ## make new unclustered matrix
        state = []
        for i in range(20):
            state.append([0])

        state[0]  = [1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0]
        state[1]  = [0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0]
        state[2]  = [1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0]
        state[3]  = [1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0]
        state[4]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[5]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[6]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[7]  = [1,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0]
        state[8]  = [0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0]
        state[9]  = [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0]
        state[10] = [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0]
        state[11] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1]
        state[12] = [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0]
        state[13] = [0,0,1,1,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0]
        state[14] = [0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,0]
        state[15] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        state[16] = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0]
        state[17] = [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0]
        state[18] = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1]
        state[19] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1]

        state = np.array(state)
        state = np.reshape(state, (-1,1))

        self.state = np.array(state)
        
        # if self.render_mode == "human":
        #     self.render()
        return np.array(self.state, dtype=np.int32), {}

    def clustering(self, state):

        # initialize weight matrix
        w = np.zeros((5,20))
        for i in range(len(w)):
            for j in range(len(w[1])):
                w[i][j] = np.random.random()
                
        no = 1
        do = 5
        T  = 300
        t  = 1
            
        # initialize result matrix
        cluster_result = np.zeros((5,20))
        n = no * (1-t/T)
        d = round(do * (1-t/T))

        # calculate Euclidean distances between each row and weight vectors
        for i in range(len(state)): # fix one row at state matrix
            distances = np.zeros(len(w))
            for j in range(len(distances)):
                dist = 0
                for k in range(len(state[1])):
                    dist += (state[i][k] - w[j][k]) ** 2
                distances[j] = dist
            idx = np.argmax(distances)
            cluster_result[idx][i] = 1

            
            # update the winning neuron
            for m in range(len(state)):
                w[idx][m] += n * (state[i][m] - w[idx][m])
            
            # update the neighbour neurons
            if idx == 0:
                w[1][m] += n * (state[i][m] - w[1][m])
                
            if idx == 1:
                w[0][m] += n * (state[i][m] - w[0][m])
                w[2][m] += n * (state[i][m] - w[2][m])

            if idx == 2:
                w[1][m] += n * (state[i][m] - w[1][m])
                w[3][m] += n * (state[i][m] - w[3][m])

            if idx == 3:
                w[2][m] += n * (state[i][m] - w[2][m])
                w[4][m] += n * (state[i][m] - w[4][m])

            if idx == 4:
                w[3][m] += n * (state[i][m] - w[3][m])
            


        a = 0.5
        b = 0.5

        # cluster_result.sum(axis=1): each row vector마다 그 성분합을 계산하라는 것
        classify_components_into_modules = cluster_result.sum(axis=1) 
        # print('classify_components_into_modules:',classify_components_into_modules)
        # classify_components_into_modules: 해당 모듈에 얼마나 많은 component가 분류되었는지, 속해 있는지 알수 있음

        sorted_classify_components_into_modules = copy.deepcopy(classify_components_into_modules)
        sorted_classify_components_into_modules.sort()
        # print('sorted_classify_components_into_modules:',sorted_classify_components_into_modules)

        reversed = sorted_classify_components_into_modules[::-1]
        # print('reversed:',reversed)

        new_order = []
        check = np.ones(shape=(20,), dtype=np.int8)
        for i in range(len(reversed)):
            for j in range(len(classify_components_into_modules)):
                if reversed[i] == classify_components_into_modules[j]:
                    for k in range(20):
                        if cluster_result[j][k] == 1 and check[k] == 1:
                            new_order.append(k)
                            check[k] = 0

        # print("new_order:",new_order) # 새롭게 바뀐 순서
        # print(len(new_order))        


        clustered_matrix = np.eye(20, dtype=np.int32)

        # make New DSM matrix
        for i in range(len(clustered_matrix)):
            for j in range(i,len(clustered_matrix)):
                if state[i][j] == 1:
                    clustered_matrix[new_order.index(i)][new_order.index(j)] = 1
                    clustered_matrix[new_order.index(j)][new_order.index(i)] = 1            



        # initialize the S_in and S_out
        S_in  = 0
        for i in range(len(classify_components_into_modules)):
            S_in += 0.5 * classify_components_into_modules[i] * (classify_components_into_modules[i] - 1)
        S_out = 0
            

        for i in range(len(state)):
            for j in range(i+1,len(state[0])):
                if state[i][j] == 1:
                    # comp_i and comp_j are in same module
                    if [row[i] for row in cluster_result] == [row[j] for row in cluster_result]: 
                        S_in -= 1
                        
                    else: # comp_i and comp_j are not in same module
                        S_out += 1
                        
                        
        CE = 1 / (a * S_in + b * S_out)

        return clustered_matrix, CE 

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False