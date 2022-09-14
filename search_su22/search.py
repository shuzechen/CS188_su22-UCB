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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def generic_search(problem, fringe, fringe_op):
    closed = set()
    start = (problem.getStartState(), 0, [])
    fringe_op(fringe, start, 0)

    while not fringe.isEmpty():
        (node, cost, path) = fringe.pop()
        if problem.isGoalState(node) :
            return path
        if not node in closed :
            closed.add(node)
            for child_node, child_action, child_cost in problem.getSuccessors(node):
                new_cost = cost + child_cost
                new_path = path + [child_action]
                new_state = (child_node, new_cost, new_path)
                fringe_op(fringe, new_state, new_cost)


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """    
    '''
    reached = set()
    stack = util.Stack()
    path = {}
    stack.push(problem.getStartState())
    path[problem.getStartState()] = []
    for frontier in problem.getSuccessors(problem.getStartState()) :
        stack.push(frontier[0])
        path[frontier[0]] = [frontier[1]]
        reached.add(frontier[0])   

    while stack.isEmpty() == False :
        tempstate = stack.pop()
        #reached.add(tempstate)
        if problem.getSuccessors(tempstate) != None :
            for frontier in problem.getSuccessors(tempstate) :
                if frontier[0] not in reached :
                    reached.add(frontier[0])
                    stack.push(frontier[0])
                    i = path[tempstate].copy()
                    #i = i.append(frontier[1])
                    path[frontier[0]] = i
                    path[frontier[0]].append(frontier[1])
                    #print(frontier[0])
                    #print(' : ')
                    #print(path[frontier[0]])
                    #print(i)
                    #print(path[frontier[0]])
                    #path[frontier[0]] = [frontier[1]]
                    if problem.isGoalState(frontier[0]) == True :
                        return path[frontier[0]]
        *** IT CAN BE SHOWN THAT AUTOGRADER IS NOT A SOLUTION TO ALL. ***
        '''
    fringe = util.Stack()
    def fringe_op(fringe, state, cost):
        fringe.push(state)
        
    return generic_search(problem, fringe, fringe_op)
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = util.Queue()
    def fringe_op(fringe, state, cost):
        # print(1)
        fringe.push(state)
        
    return generic_search(problem, fringe, fringe_op)

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    fringe = util.PriorityQueue()
    def fringe_op(fringe, state, cost):
        fringe.push(state, cost)
        
    return generic_search(problem, fringe, fringe_op)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()
    def fringe_op(fringe, state, cost):
        new_cost = cost + heuristic(state[0], problem)
        fringe.push(state, new_cost)
        
    return generic_search(problem, fringe, fringe_op)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch