# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def Minimax(self, gameState, currentdepth):
        totalagents = gameState.getNumAgents()     # Get total number of agents
        agentid = currentdepth % totalagents - 1     # Calculate current agent index
        moves = gameState.getLegalActions(agentid)     # Get legal moves for the current agent
        values = []     # Initialize list to store move values
        for move in moves:      # Iterate over legal moves
            successorstate = gameState.generateSuccessor(agentid, move)        # Generate successor state for the move
            if (successorstate.isLose() or successorstate.isWin() or currentdepth == self.depth * totalagents):       # Base case: check if state is a win/lose or maximum depth reached
                values.append((move, self.evaluationFunction(successorstate)))      # Append move and its score
            else:
                values.append((move ,self.Minimax(successorstate, currentdepth + 1)[1]))       # Recursively call Minimax for successor state and append move and score
        if (agentid == 0):     # If current agent is the AI (maximizer)
            bestmove = max(values, key=lambda pair: pair[1])      # Choose move with maximum value
        else:       # If current agent is an opponent (minimizer)
            bestmove = min(values, key=lambda pair: pair[1])      # Choose move with minimum value
        return bestmove       # Return the best move and its value

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.Minimax(gameState, 1)[0]        # Start Minimax with depth 1 and return best move
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def Minimax(self, gameState, depth, alpha, beta):
        totalagents = gameState.getNumAgents()     # Get total number of agents in the game
        agentid = depth % totalagents - 1     # Calculate current agent's index
        values = []     # Initialize list to store values of moves
        actions = gameState.getLegalActions(agentid)       # Get legal actions for the current agent
        if(agentid == 0):      # If the current agent is the AI (maximizer)
            for action in actions:      # Iterate through all possible actions
                successorstate = gameState.generateSuccessor(agentid, action)      # Generate successor state for an action
                if(successorstate.isLose() or successorstate.isWin() or depth == self.depth * totalagents):        # Check if the state is terminal or depth limit is reached
                    v = self.evaluationFunction(successorstate)     # Evaluate the terminal state
                else:
                    v = self.Minimax(successorstate, depth + 1, alpha, beta)[1]     # Recursively call Minimax for successor state
                values.append((action, v))      # Append action and its value
                if(v > beta): break     # Alpha-beta pruning (cut off if value exceeds beta)
                alpha = max(alpha, v)       # Update alpha if necessary
            bestmove = max(values, key=lambda x: x[1])        # Choose the action with the maximum value
        else:       # If the current agent is an opponent (minimizer)
            for action in actions:      # Iterate through all possible actions
                successorstate = gameState.generateSuccessor(agentid, action)      # Generate successor state for an action
                if (successorstate.isLose() or successorstate.isWin() or depth == self.depth * totalagents):       # Check if the state is terminal or depth limit is reached
                    v = self.evaluationFunction(successorstate)     # Evaluate the terminal state
                else:
                    v = self.Minimax(successorstate, depth + 1, alpha, beta)[1]     # Recursively call Minimax for successor state
                values.append((action, v))      # Append action and its value
                if (v < alpha): break       # Alpha-beta pruning (cut off if value is less than alpha)
                beta = min(beta, v)     # Update beta if necessary
            bestmove = min(values, key=lambda x: x[1])        # Choose the action with the minimum value    
        return bestmove       # Return the best action and its value for the current agent
        
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.Minimax(gameState, 1, -99999999, 99999999)[0]       # Start Minimax with initial alpha and beta values and return the best action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
