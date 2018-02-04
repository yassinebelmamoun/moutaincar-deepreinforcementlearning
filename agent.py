import numpy as np
import math

"""
Contains the definition of the agent that will run in an
environment.
"""

class UltimateAgent:
    """ My Agent """
    def __init__(self):
        """Init a new agent"""
        # Parameters
        self.learning_rate = 0.01
        self.gamma = 1
        self.epsilon = 0.5

        self.d_min = -150
        self.actions = [-1, 0, 1]

        # Segmentation of x and vx
        self.p = 30
        self.k = 30

        model_x_vx = np.zeros((self.p, self.k), type=np.int),
        self.w_table = {
                -1: model_x_vx,
                0 : model_x_vx,
                1 : model_x_vx,
            }

    def Q_compute(self, x, vx, a):
        result = 0
        for i in range(self.p + 1):
            for j in range(self.k + 1):
                result += self.w_table[a][i][j] * self.phi_compute(i, j, x, vx)

    def phi_compute(self, i, j, x, vx):
        """ Compute the value of phi[i, j](x, vx) """
        return math.exp(-(x - self.s_x(i))) * math.exp(-(vx - self.s_vx(j)))

    def s_x(self, i):
        """ Compute the first value (position) of s[i,j] """
        return (-1) * self.d_min + i * (self.d_min / self.p)

    def s_vx(self, j):
        """ Compute the second value (vitesse) of s[i,j] """
        return (-20) + j * (40 / self.k)

    def reset(self, x_range):
        self.d_min = abs(x_range[0])

    def act(self, observation):
        x, vx = observation
        if np.random.random() > self.epsilon:
            action = np.random.choice([-1, 0, 1])
        else:
            action = np.random.choice([-1, 0, 1])
        return action

    def reward(self, observation, action, reward):
        # How to learn W_table ? Gradient Descent .. but how?
        pass

class SimpleAgent:
    def __init__(self):
        """ Init """
        self.learning_rate = 0.1
        self.gamma = 1
        self.p = 50
        self.k = 20 
        self.d_min = -150
        self.q_table = {} # (x,vx) is the key and value is initialized to 0
        self.epsilon = 0.5
        self.DEFAULT_VALUE_Q_TABLE_INIT = 0
        self.last_observation = False

    def get_ax(self, x):
        """ Approximate position """
        return round(self.p * (x - self.d_min) / abs(self.d_min), 0)

    def get_avx(self, vx):
        """ Approximate speed """
        return round(self.k * (vx + 20) / 40, 0)

    def get_appr(self, observation):
        """ Return approximation of (position, speed) """
        x, vx = observation
        return (self.get_ax(x), self.get_avx(vx))

    def learn(self, last_observation, new_observation, action, reward):
        """ Learn Q-table """
        last_observation = self.get_appr(last_observation)
        new_observation = self.get_appr(new_observation)
        if not (last_observation, action) in self.q_table:
            self.q_table[(last_observation, action)] = 0 # self.DEFAULT_VALUE_Q_TABLE_INIT
        if not (new_observation, action) in self.q_table:
            self.q_table[(new_observation, action)] = 0 # self.DEFAULT_VALUE_Q_TABLE_INIT

        self.q_table[(last_observation, action)] += self.learning_rate * ( \
                                      reward + self.gamma * self.get_max_q_table(new_observation) \
                                      - self.q_table[(last_observation, action)])

    def get_max_q_table(self, observation):
        """ Return maximum value of q-table for an observation """
        q_table_ob_act = [self.q_table[(observation, action)] for action in [-1, 0, 1] if (observation, action) in self.q_table]
        if not(q_table_ob_act):
            return 0
        else:
            return max(q_table_ob_act)

    def get_action_max_q_table(self, observation):
        """ Return action for observation with maximum value """
        q_table_ob_act = [self.q_table[(observation, action)] for action in [-1, 0, 1] if (observation, action) in self.q_table]
        if not(q_table_ob_act):
            print('RANDOM')
            return np.random.choice([-1,0,1])
        max_value = max(q_table_ob_act)
        max_actions = [action for action in [0, -1, 1] if (observation, action) in self.q_table and \
                      max_value == self.q_table[(observation, action)] and \
                      (observation, action) in self.q_table]
        return np.random.choice(max_actions)

    def act(self, observation):
        """ Act based on epsilon-greedy """
        observation = self.get_appr(observation)
        action = self.get_action_max_q_table(observation)
        print('observation/action: {} {}'.format(observation, action))
        return action

    def reset(self, x_range):
        self.d_min = x_range[0]
        self.last_observation = False

    def reward(self, observation, action, reward):
        if self.last_observation is not False:
            last_observation = self.last_observation
            new_observation = observation
            self.learn(last_observation, new_observation, action, reward)
            self.last_observation = observation
        self.last_observation = observation

class QLearningAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1

        self.p = 30
        self.k = 30 
        self.s = np.zeros((self.p+1, self.k+1, 2))

        self.W_minus1 = np.zeros((self.p+1, self.k+1))
        self.W_0 = np.zeros((self.p+1, self.k+1))
        self.W_1 = np.zeros((self.p+1, self.k+1))

        self.alpha = 0.1
        self.gamma = 1
        self.observation_old = None
        self.action_old = None
        self.reward_old = None


    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change when starting a new
        episode of the same game, but your initial location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        for i in range(self.p+1):
            for j in range(self.k+1):
                self.s[i,j,0] = x_range[0] + i*(x_range[1] - x_range[0])/self.p
                self.s[i,j,1] = -20 + j*40/self.k


    def phi(self, observation):
        x = observation[0]
        vx = observation[1]

        phi = np.zeros((self.p+1,self.k+1))

        for i in range(self.p+1):
            for j in range(self.k+1):
                phi[i,j] = np.exp(-((x-self.s[i,j,0])**2)/self.p) * np.exp(-((vx-self.s[i,j,1])**2)/self.k)

        return phi


    def Q(self, phi, action):
        if action == -1:
            return sum(sum(self.W_minus1*phi))
        elif action == 0:
            return sum(sum(self.W_0*phi))
        else:
            return sum(sum(self.W_1*phi))


    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if self.observation_old is not None:

            phi = self.phi(observation)
            max_a = max([self.Q(phi,-1) , self.Q(phi,0) , self.Q(phi,1)])

            phi_o = self.phi(self.observation_old)
            correction = self.reward_old + self.gamma * max_a - self.Q(phi_o,self.action_old)

            if self.action_old == -1:
                self.W_minus1 = self.W_minus1 + self.alpha * correction * phi_o


            elif self.action_old == 0:
                self.W_0 = self.W_0 + self.alpha * correction * phi_o


            else:
                self.W_1 = self.W_1 + self.alpha * correction * phi_o

        phi_new = self.phi(observation)
        ## Choose the act with an epsilon greedy strategy
        if np.random.random() > self.epsilon and self.observation_old is not None:
            return np.argmax([self.Q(phi_new,-1) , self.Q(phi_new,0) , self.Q(phi_new,1)]) - 1
        else:
            return np.random.choice([-1, 0, 1])


    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """

        self.observation_old = observation
        self.action_old = action
        self.reward_old = reward




class TDAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.epsilon = 0.1

        self.lambdaa = 0.99
        self.p = 40 #100
        self.k = 20 #40
        self.s = np.zeros((self.p+1,self.k+1,2))


        self.W_minus1 = np.random.rand(self.p+1,self.k+1)
        self.W_0 = np.random.rand(self.p+1,self.k+1)
        self.W_1 = np.random.rand(self.p+1,self.k+1)

        self.trace_minus1 = np.zeros((self.p+1,self.k+1))
        self.trace_0 = np.zeros((self.p+1,self.k+1))
        self.trace_1 = np.zeros((self.p+1,self.k+1))



        self.nbr_plays = 0


        self.alpha = 0.1
        self.gamma = 1
        self.observation_old = None
        self.action_old = None
        self.reward_old = None

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change when starting a new
        episode of the same game, but your initial location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        for i in range(self.p+1):
            for j in range(self.k+1):
                self.s[i,j,0] = x_range[0] + i*(x_range[1] - x_range[0])/self.p
                self.s[i,j,1] = -20 + j*40/self.k



    def phi(self, observation):
        x = observation[0]
        vx = observation[1]

        phi = np.zeros((self.p+1,self.k+1))

        for i in range(self.p+1):
            for j in range(self.k+1):
                phi[i,j] = np.exp(-((x-self.s[i,j,0])**2)/self.p) * np.exp(-((vx-self.s[i,j,1])**2)/self.k)

        return phi



    def Q(self,observation, action):
        if action == -1:
            return sum(sum(self.W_minus1*self.phi(observation)))
        elif action == 0:
            return sum(sum(self.W_0*self.phi(observation)))
        else:
            return sum(sum(self.W_1*self.phi(observation)))


    def act(self, observation):
        ## Update weight matrices
        if self.observation_old is not None:

            if self.action_old == -1:
                 self.trace_minus1 = self.gamma * self.lambdaa * self.trace_minus1 + self.phi(self.observation_old)
                 delta = self.reward_old + self.gamma * self.Q(observation,-1) - self.Q(self.observation_old,-1)
                 self.W_minus1 = self.W_minus1 + self.alpha * delta * self.trace_minus1

            elif self.action_old == 0:
                 self.trace_0 = self.gamma * self.lambdaa * self.trace_0 + self.phi(self.observation_old)
                 delta = self.reward_old + self.gamma * self.Q(observation,0) - self.Q(self.observation_old,0)
                 self.W_0 = self.W_0 + self.alpha * delta * self.trace_0

            else:
                 self.trace_1 = self.gamma * self.lambdaa * self.trace_1 + self.phi(self.observation_old)
                 delta = self.reward_old + self.gamma * self.Q(observation,0) - self.Q(self.observation_old,0)
                 self.W_1 = self.W_1 + self.alpha * delta * self.trace_1



        if np.random.random() > self.epsilon and self.observation_old is not None:
            return np.argmax([self.Q(observation,-1) , self.Q(observation,0) , self.Q(observation,1)]) - 1
        else:
            return np.random.choice([-1, 0, 1])



    def reward(self, observation, action, reward):
        self.observation_old = observation
        self.action_old = action
        self.reward_old = reward

Agent = TDLearningAgent


