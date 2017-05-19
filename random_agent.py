import cv2
import random
import matplotlib.pylab as plt
import numpy as np
import scipy.stats as stats
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState


class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        # Add member variables to your class here
        self.total_reward = 0
	self.peak_reward = 0
	self.total_rewards_dict = {}
	self.peak_rewards_dict = {}
	self.episodes = 100

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        # Reset the total reward for the episode
        self.total_reward = 0

	# Reset the peak reward for the episode
        self.peak_reward = 0

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
	
	"""The method simply uses the randint method from the random package
	   to come up with a number from 1 to 4, which is then used to select
	   one of the four available actions."""
	move = random.randint(1,4)
	
	if move == 1:
		self.total_reward += self.move(Action.ACCELERATE)
		#print "accelerate"	
	elif move == 2:
		self.total_reward += self.move(Action.BREAK)
		#print "break"
	elif move == 3:
		self.total_reward += self.move(Action.LEFT)
		#print "left"
	elif move == 4:
		self.total_reward += self.move(Action.RIGHT)
		#print "right"

	# Updating maximum reward value
	if self.total_reward > self.peak_reward: 
		self.peak_reward = self.total_reward

    def sense(self, grid):
        """ Constructs the next state from sensory signals.

        gird -- 2-dimensional numpy array containing the latest grid
                representation of the environment
        """
        # Visualise the environment grid
        cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        """ Performs the learning procudre. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        pass

    def callback(self, learn, episode, iteration):
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
	self.total_rewards_dict[episode] = self.total_reward
	self.peak_rewards_dict[episode] = self.peak_reward
        print "{0}/{1}: {2}".format(episode, iteration, self.total_reward)
        # Show the game frame only if not learning
        if not learn:
            cv2.imshow("Enduro", self._image)
            #cv2.waitKey(1) #Disabled to speed things up
	

if __name__ == "__main__":
    a = RandomAgent()

    #Perform one hundred episodes
    a.run(False, episodes=a.episodes, draw=False)
    
    #print 'Total reward for run ' + str(i) + ' is ' + str(a.total_reward)

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
