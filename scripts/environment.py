
# Importing simulation
from  drive.MyDrive.racecar.simulation import python_env

# Importing gym environment
import gym
from gym import spaces

# Other imports
import sys
import random
import numpy as np
import time
from math import pi, sqrt, atan2, cos, sin

# Defining helper functions wrt. reward shaping

def wallhit(scan):
    thresh = 0.30
    return min(scan) < thresh

def goalhit(dist):
    thresh = 0.1
    return dist < thresh

# def middle(scan):
    
#     if (scan[3]+scan[4]+scan[5])/3-(scan[24]+scan[25]+scan[26])/3<=0.3:
#       return True
#     elif (1.2>=(scan[3]+scan[4]+scan[5])/3>=0.5) or (1.2>=(scan[23]+scan[24]+scan[25])/3>=0.5):
#       return True
#     else:
#       return False


def closer(dist1, dist2):
    return dist2 < dist1

def diff(a1, a2):
    """
    Calculates the difference between two angles as well as the sign from a2 to
    a1.
    """
    phi = abs(a1 - a2) % (2 * pi)
    test = ((a1 - a2 >= 0) & (a1 - a2 <= pi)) or ((a1 - a2 <= -pi) & (a1 - a2 >= -2 * pi))

    if test:
        sign = 1
    else:
        sign = -1

    if phi > pi:
        return 2 * pi - phi, sign
    else:
        return phi, sign

class RacecarEnv(gym.Env):
    """
    Custom Environment that follows gym interface and defines the MIT-racecar
    simulation environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, turns):
        super(RacecarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(1)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

        # Defining the maximum distance to goal (used to normalize)
        self.maxgoaldist = 5

        self.turns = turns

        self.sum_dist = 0

        self.num_checkpoint = 0
        
        self.reward_2=0

        # Spawning the simulation
        self.sim = python_env(turns, 0)

        # position record
        self.position = []
    def observe(self):
        dist = sqrt((self.goal[0] - self.carx)**2 + (self.goal[1] - self.cary)**2)
        angle_to_goal = atan2((self.goal[1] - self.cary), (self.goal[0] - self.carx))
        phi, sign = diff(self.cartheta, angle_to_goal)
        return dist, phi, sign

    def checkpointhit(self, dist_turn, length):

        self.sum_dist += dist_turn
        dist_checkpoint = sqrt((self.sim.car[0]-self.sim.points[self.num_checkpoint+1][0])**2+(self.sim.car[1]-self.sim.points[self.num_checkpoint+1][1])**2)


        if (abs(self.sum_dist) > length[self.num_checkpoint] or dist_checkpoint <= 0.7  ):
        #if (dist_checkpoint <= 0.5  ):

            # self.sum_dist = 0
            return True
        else:
            return False


    def step(self, action):
        '''
        Defining the step function - returning the observation space, reward and
        whether or not the action caused a reset (hitting goal or wall)
        '''

        # Calculating previous distances and angles
        old_dist, old_phi, old_sign = self.observe()
        self.oldx, self.oldy = self.carx, self.cary

        self.sim.action(action)

        # Fetching the resulting lidar as well as postion data
        sensor = self.sim.lidar()
        self.carx, self.cary, self.cartheta = self.sim.car
        # Add position
        self.position.append([self.carx, self.cary])
        # Calculating current distances and angles
        dist, phi, sign = self.observe()

        # Defining the observation vector
        self.obs = list(1 - np.array(sensor) / self.maxgoaldist)
        act = action[0]
        self.obs.extend([dist / self.maxgoaldist, phi / pi, sign])
        self.obs.extend(act)

        # Incrementing step counter
        self.current_step += 1

        #calculate how many checkpoint has been hit
        angle_move = atan2((self.cary-self.oldy),(self.carx-self.oldx))
        angle = angle_move-self.sim.angles[self.num_checkpoint]
        dist_turn = sqrt((self.cary-self.oldy)**2+(self.carx-self.oldx)**2)*cos(angle) #### cos

   
        if self.checkpointhit(dist_turn, self.sim.lens):

            #print("I HIT ONE POINT")
            if (self.sim.points[self.num_checkpoint+1] !=self.goal ):

                self.num_checkpoint += 1
            #print("                ", self.num_checkpoint, self.sim.points[self.num_checkpoint], [self.carx, self.cary],self.sum_dist)
                self.sum_dist = 0
       
        # if middle(sensor):

        #     self.reward_2= -1
        # else:
        #     self.reward_2=0


        if self.num_checkpoint ==0:
           reward_1 = sqrt((self.sim.car[0]-self.sim.points[self.num_checkpoint][0])**2+(self.sim.car[1]-self.sim.points[self.num_checkpoint][1])**2)

        else:
           reward_1 = sum(self.sim.lens[: self.num_checkpoint]) + sqrt((self.sim.car[0]-self.sim.points[self.num_checkpoint][0])**2+(self.sim.car[1]-self.sim.points[self.num_checkpoint][1])**2)
             
             
             

        if wallhit(sensor):
            print('I HIT A WALL!', dist)

            reward_wall = 400-2*reward_1
            #reward_wall = 99


            return self.obs,reward_wall, True, {}

        if goalhit(dist):
            print('I HIT THE GOAL!')

            reward_goal = -1000-2*reward_1 #-100  0.5
            #reward_goal = 0

            return self.obs, reward_goal, True, {}

        # if (self.sim.points[self.num_checkpoint] == self.goal and not goalhit(dist)) :

        #     print ('I FAIL',dist)

        #     reward_fail = 0-reward_1

        #     return self.obs, reward_fail, True, {}

        # Defining the reward shaping
        #reward = 2+ self.reward_2
        reward = 1

        return np.array(self.obs).reshape(1, -1), reward, False, {}

    def reset(self, seed):
        """
        Resetting the environment for a new run
        """
        random.seed(seed)
        np.random.seed(seed)
        self.num_checkpoint = 0
        self.sum_dist = 0
        self.sim = python_env(self.turns, seed)
        #print("point", self.sim.points)
        #print("lens", self.sim.lens)

        # Initializing the goal
        self.goal = self.sim.goal

        self.sim.spawn(0, 0, 0)

        # Initializing step counter
        self.current_step = 0

        # Recieving sensor information
        sensor = self.sim.lidar()
        self.carx, self.cary, self.cartheta = self.sim.car

        # Calculating distances and angles for the observation vector
        dist, phi, sign = self.observe()

        # Defining the observation vector
        self.obs = list(1 - np.array(sensor) / self.maxgoaldist)
        act = [0, 0]
        self.obs.extend([dist / self.maxgoaldist, phi / pi, sign])
        self.obs.extend(act)

        return np.array(self.obs).reshape(1, -1)
