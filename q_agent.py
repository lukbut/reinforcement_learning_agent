import cv2
import numpy as np
import matplotlib.pylab as plt
import random
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState

import sys

class QAgent(Agent):
    def __init__(self):
        super(QAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
	self.episodes = 300 # Number of episodes
	self.alpha = 0.1 # Alpha, for learning equation representation
	self.gamma = 0.9 # Gamma, for learning equation representation 
	self.epsilon = 1.0 # Epsilon, for greedy learning rate 
	self.decay_rate = 5000.0 # Epsilon greedy decay rate
	self.total_rewards_dict = {}
	self.peak_rewards_dict = {}
	
	#Initialising the q_values to an initial value of zero, for the two states
	print 'Initialise Q(s,a)'
	self.q_values = np.zeros((4,4))

	# Reset the maximum reward for the episode
	self.peak_reward = 0

	# Reset the action count
	self.action_count = 0

	print 'Repeat many times'

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
	# Add total and peak to dictionary from previous episode

        # Reset the total reward for the episode
        self.total_reward = 0

	# Reset the current reward for the episode
	self.current_reward = 0

        # Setting the current grid to be the initial state of the grid
	self.grid = grid

	# Checking what the initial state is
	self.current_state = self.checkstate(grid)
	print 'Initial state is ' + str(self.current_state)	

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        # You can get the set of possible actions and print it with:
        # print [Action.toString(a) for a in self.getActionsSet()]

        # Execute the action and get the received reward signal
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BREAK
        # Do not use plain integers between 0 - 3 as it will not work
        # self.total_reward += self.move(action)

	# Reset the current reward for each action
	self.current_reward = 0

	print 'NEW ACTION no.' + str(self.action_count)
	random_int = np.random.random()
	print 'Random integer is ' + str(random_int)

	# Epsilon greedy logic, to balance exploration and exploitation in a decaying fashion
	if random_int > self.epsilon: 
		# Picking the optimum q value in the respective state row, exploiting
		max_q = np.nanmax(self.q_values[self.current_state,:])
	
		print 'maximum q value is ' + str(max_q) + ' for state ' + str(self.current_state)

		# Picking the action depending on the position of the column of the optimum q value
		max_q_index = np.where(self.q_values[self.current_state,:]==max_q)

		print 'maximum q value is at index' + str(max_q_index)

		if len(max_q_index[0]) > 1:
			#If more than one optimum action found, break tie arbitrarily
			self.move_int = max_q_index[0].item(random.randint(0,len(max_q_index[0])-1))
			print "random greedy action selected " + str(self.move_int)
		else: 
			#Perform optimum action greedily
			self.move_int = max_q_index[0].item(0)
			print "sole optimum greedy action selected "

	else:
		# Explore randomly by picking a random action
		self.move_int = random.randint(0,3)
		print "random exploratory action selected"

	#Performing action accoridng to the greedy epsilon choice

	if self.move_int == 0:
		self.current_reward += self.move(Action.ACCELERATE)
		self.total_reward += self.current_reward
		print "ACTION: accelerate"	
	elif self.move_int == 1:
		self.current_reward += self.move(Action.BREAK)
		self.total_reward += self.current_reward
		print "ACTION: break"
	elif self.move_int == 2:
		self.current_reward += self.move(Action.LEFT)
		self.total_reward += self.current_reward
		print "ACTION: left"
	elif self.move_int == 3:
		self.current_reward += self.move(Action.RIGHT)
		self.total_reward += self.current_reward
		print "ACTION: right"
	
	# printing current reward
	print 'Reward ' + str(self.current_reward)

	# Updating maximum reward value
	if self.total_reward > self.peak_reward: 
		self.peak_reward = self.total_reward

	# Incrementing the action count
	self.action_count += 1

	# Decaying epsilon between action
	self.epsilon = self.decay_rate/(self.action_count + self.decay_rate)

	print 'New epsilon value is ' + str(self.epsilon)

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))
	
	# Storing previous grid and reading new grid
	self.previousGrid = self.grid
	self.grid = grid

	# Storing previous state
	self.previous_state = self.current_state

	print 'Previous state was ' + str(self.previous_state)
	
	# Checking what the new state is 
	self.current_state = self.checkstate(grid)

	print 'New state is ' + str(self.current_state)
	
    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        # This is where the q values are going to be updated 
	# s is represented by self.previous_state
	# s' is represented by self.current_state
	# a is represented by self.move_int
	# r is represented by self.current_reward
	# alpha is represented by self.alpha
	# gamma is represented by self.gamma 
	# maxaQ(s',a') is represented by max_a

	print 'action was ' + str(self.move_int)
	print 'reward was ' + str(self.current_reward)
	print 'alpha is ' + str(self.alpha)
	print 'gamma is ' + str(self.gamma)
	
	# Calculating the maxaQ(s',a')
	max_a_q = max(self.q_values[self.current_state,:])

	print 'max a q is ' + str(max_a_q)

	# Calculating the new value as per the q-learning algorithm specification
 	new_q_value = self.q_values[self.previous_state,self.move_int] + (self.alpha*(self.current_reward + (self.gamma*max_a_q) - self.q_values[self.previous_state,self.move_int]))

	print 'updating value of state ' + str(self.previous_state) + ' at move ' + str(self.move_int) + ' to value ' + str(new_q_value)

	#if self.current_reward <> 0: 
	#	raw_input("Value not 0. Press enter to continue")
	
	# Updating value in q matrix
	self.q_values[self.previous_state,self.move_int] = new_q_value

	print 'Q Values are now'
	print self.q_values

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        print "{0}/{1}: T {2} P {3} C {4}".format(episode, iteration, self.total_reward, self.peak_reward, self.current_reward)

	self.total_rewards_dict[episode] = self.total_reward
	self.peak_rewards_dict[episode] = self.peak_reward
        # Show the game frame only if not learning
        """if not learn:
            cv2.imshow("Enduro", self._image)
            cv2.waitKey(40)
	"""
	#cv2.imshow("Enduro", self._image)
        #cv2.waitKey(1)
	#raw_input("End of action. Press enter to continue")

    def checkstate(self, grid):
	# State 0 = opponent directly in front of agent
	# State 1 = opponents left of agent
	# State 2 = opponents right of agent
	# State 3 = clear road ahead

	print grid

	agent_index = np.where(grid==2)
	
	# Obtaining position of agent on grid
	agent_cell = agent_index[1].item(0)

	# Obtaining left and right values
	left = agent_cell - 4

	if left < 0:
		left = 0
	
	right = agent_cell + 4
	
	if right > 10:
		right = 10
			
	# Checking the value in front of agent on grid
	in_front_agent = grid[:,agent_cell]
	left_agent = grid[:3,left:agent_cell]
	right_agent = grid[:3,agent_cell+1:right]

	print 'left' + str(left)
	print 'right' + str(right)

	print in_front_agent
	print left_agent
	print right_agent

	opponents = np.where(in_front_agent==1)
	opponents_left = np.where(left_agent==1)
	opponents_right = np.where(right_agent==1)
	
	print ' opponents_front at '
	print opponents

	print ' opponents_left at '
	print opponents_left

	print ' opponents_right at '
	print opponents_right

	if len(opponents[0]) > 0:
		#raw_input("opponents_front. Press enter to continue")
		return 0
	elif len(opponents_left[0]) > 0:
		#raw_input("opponents_left. Press enter to continue")
		return 1
	elif len(opponents_right[0]) > 0:
		#raw_input("opponents_right. Press enter to continue")
		return 2
	else:
		#raw_input("No opponents. Press enter to continue")
		return 3

if __name__ == "__main__":
    a = QAgent()
    a.run(True, episodes=a.episodes, draw=True)
    #print 'Total reward: ' + str(a.total_reward)
    #print 'Peak reward: ' + str(a.peak_reward)

    print 'Total rewards'
    for x in a.total_rewards_dict:
    	print x, a.total_rewards_dict[x]

    print 'Peak rewards'
    for x in a.peak_rewards_dict:
    	print x, a.peak_rewards_dict[x]

    #Showing the results plot
    plt.plot(*zip(*sorted(a.total_rewards_dict.items())))
    plt.xticks(a.total_rewards_dict.keys())
  
    fig = plt.show()
	
    #Showing the distribution plot
    plt.hist(a.total_rewards_dict.values())
    fig2 = plt.show()

    #Showing the peak results plot
    plt.plot(*zip(*sorted(a.peak_rewards_dict.items())))
    plt.xticks(a.peak_rewards_dict.keys())
  
    fig3 = plt.show()

    #extracting values into numpy array
    values = np.fromiter(iter(a.total_rewards_dict.values()), dtype=float)

    #Calculating and printing the mean
    print 'Mean: ' + str(np.mean(values))

    #Calculating and printing the variance
    print 'Variance: ' + str(np.var(values))

