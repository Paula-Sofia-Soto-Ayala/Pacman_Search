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
        util.raise_method_not_defined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_method_not_defined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_method_not_defined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_method_not_defined()


def tinyMazeSearch(problem: SearchProblem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
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
    "*** YOUR CODE HERE ***"
    # Obtiene el estado inicial del problema
    start_state = problem.getStartState()
    # Crea una pila vacía para almacenar los estados a visitar
    stack = util.Stack()
    # Apila el estado inicial
    stack.push(start_state)

    # Crea un diccionario vacío para almacenar el camino inverso
    back_track = {}

    # Crea un conjunto vacío para almacenar los estados visitados
    states_visited = {start_state}

    # Crea una variable para almacenar el estado objetivo
    goal_state = None

    # Repite hasta que la pila esté vacía o se encuentre el estado objetivo
    while not stack.isEmpty() and goal_state is None:
      # Desapila el último estado de la pila
      curr_state = stack.pop()

      # Obtiene los sucesores del estado actual
      successors = problem.getSuccessors(curr_state)
      # Recorre los sucesores del estado actual
      for next_state, action, _ in successors:
        # Si el sucesor ya está visitado, lo ignora
        if next_state in states_visited:
          continue
        # Añade el sucesor al conjunto de visitados
        states_visited.add(next_state)
        # Apila el sucesor
        stack.push(next_state)
        # Guarda el camino inverso desde el sucesor al estado actual y la acción realizada
        back_track[next_state] = (curr_state, action)

      # Si el estado actual es el objetivo, lo guarda y termina el bucle
      if problem.isGoalState(curr_state):
        goal_state = curr_state
        break

    # Si se encontró el estado objetivo, reconstruye el camino desde el inicio hasta el objetivo
    if goal_state is not None:
      # Crea una cola vacía para almacenar las acciones
      actions = util.Queue()
      # Asigna el estado actual al objetivo
      curr_state = goal_state
      # Repite hasta que el estado actual sea el inicial
      while curr_state != start_state:
        # Obtiene el estado anterior y la acción desde el diccionario de camino inverso
        curr_state, action = back_track[curr_state]
        # Encola la acción al principio de la cola
        actions.push(action)

      # Devuelve la lista de acciones
      return actions.list

    # Si no se encontró el estado objetivo, devuelve una lista vacía
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the nearest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    # Crea una cola vacía para almacenar los estados a visitar
    queue = util.Queue()
    # Encola el estado inicial
    queue.push(start_state)

    # Crea un diccionario vacío para almacenar el camino inverso
    back_track = {}

    # Crea un conjunto vacío para almacenar los estados visitados
    states_visited = {start_state}

    # Crea una variable para almacenar el estado objetivo
    goal_state = None

    # Repite hasta que la cola esté vacía o se encuentre el estado objetivo
    while not queue.isEmpty() and goal_state is None:
      # Desencola el primer estado de la cola
      curr_state = queue.pop()

      # Obtiene los sucesores del estado actual
      successors = problem.getSuccessors(curr_state)
      # Recorre los sucesores del estado actual
      for next_state, action, _ in successors:
        # Si el sucesor ya está visitado, lo ignora
        if next_state in states_visited:
          continue
        # Añade el sucesor al conjunto de visitados
        states_visited.add(next_state)
        # Encola el sucesor
        queue.push(next_state)
        # Guarda el camino inverso desde el sucesor al estado actual y la acción realizada
        back_track[next_state] = (curr_state, action)

      # Si el estado actual es el objetivo, lo guarda y termina el bucle
      if problem.isGoalState(curr_state):
        goal_state = curr_state
        break

    # Si se encontró el estado objetivo, reconstruye el camino desde el inicio hasta el objetivo
    if goal_state is not None:
      # Crea una pila vacía para almacenar las acciones
      actions = util.Queue()
      # Asigna el estado actual al objetivo
      curr_state = goal_state
      # Repite hasta que el estado actual sea el inicial
      while curr_state != start_state:
        # Obtiene el estado anterior y la acción desde el diccionario de camino inverso
        curr_state, action = back_track[curr_state]
        # Apila la acción
        actions.push(action)

      # Devuelve la lista de acciones
      return actions.list

    # Si no se encontró el estado objetivo, devuelve una lista vacía
    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raise_method_not_defined()


def nullHeuristic(state, problem: SearchProblem = None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def manhattanHeuristic(position, problem, info={}):
    """The Manhattan distance heuristic for a PositionSearchProblem"""
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def aStarSearch(problem: SearchProblem, heuristic = nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    cost_dict = { start_state: 0 }

    # use the priority queue without function queue.push((start_state, [], 0), 0) # (state, actions, cost), priority
    queue = util.PriorityQueue()
    # Diccionario para reconstruir el camino tomado
    back_track = {}

    # Iniciamos el queue con el estado inicial y prioridad 0
    queue.push(start_state, 0)
    states_visited = {start_state}
    goal_state = None

    while not queue.isEmpty() and goal_state is None:
        parent = queue.pop()
        children = problem.getSuccessors(parent)

        for next_state, action, step_cost in children:
            # Si ya visitamos el estado entonces nos lo saltamos
            if next_state in states_visited:
                continue
            
            # Calcular G(N) donde N es el hijo actual y g(n)
            # es el costo de llegar del inicio al hijo
            cost_parent = cost_dict[parent]
            curr_cost = step_cost + cost_parent

            # Calcular H(N) donde N es el hijo actual y h(n)
            # es el costo heuristico del hijo al final
            curr_heur = heuristic(next_state, problem)

            # Guarda el costo total hasta el nodo actual
            cost_dict[next_state] = curr_cost

            # Calcula la prioridad / f(n)
            priority = curr_cost + curr_heur
            queue.update(next_state, priority)
            
            # Guarda el hijo en los estados visitados
            states_visited.add(next_state)

            # Guarda el camino inverso desde el sucesor al estado actual y la acción realizada
            back_track[next_state] = (parent, action)

        if problem.isGoalState(parent):
            goal_state = parent
            break

    if goal_state is not None:
        actions = util.Queue()
        parent = goal_state
        while parent != start_state:
            parent, action = back_track[parent]
            actions.push(action)

        return actions.list
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
