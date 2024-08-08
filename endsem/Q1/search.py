# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def uniformCostSearch(problem: SearchProblem):

	# Initialise priority queue
	queue = util.PriorityQueue()
	
	# Initialise visited list
	visited = list()

	# Push start state with empty list of actions and zero initial path cost into priority queue
	queue.push((problem.getStartState(), []), 0)
	
	# Run while loop until priority queue is empty
	while not queue.isEmpty():
	
		# Pop the priority queue and store the state and actions list returned
		current, path = queue.pop()
		
		# Check edge case if returned state is goal state or not
		if problem.isGoalState(current):
			
			# Return the action path list if returned state is goal state
			return path
			
		# Check edge case if returned state is already visited or not
		if current not in visited:
		
			# Add returned state to list of visited states if it is not already visited
			visited.append(current)
			
			# Find all successors of the returned state with next corresponding action and cost
			for successor, action, cost in problem.getSuccessors(current):
			
				# Push all successors into priority queue with new action list and path cost
				queue.push((successor, path + [action]), problem.getCostOfActions(path) + cost)

	util.raiseNotDefined()


# function call
ucs = uniformCostSearch

