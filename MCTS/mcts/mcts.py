from __future__ import division

import time
import math
import random


def randomPolicy(state):
    while not state.isTerminal():
        try:
            #print("begin getPossibleActions")
            action = random.choice(state.getPossibleActions())
            #print("over getPossibleActions")
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy, epsilon=0.1):
        self.epsilon = epsilon

        if timeLimit is not None:
            if iterationLimit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        #print("begin executeRound")
        node = self.selectNode(self.root)
        #print("over selectNode,begin rollout")
        reward = self.rollout(node.state)
        #print("over rollout,behgin backpropogate")
        self.backpropogate(node, reward)
        #print("over backpropogate,over executeRound")

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        #import time
        #time_start = time.time()
        #
        actions = node.state.getPossibleActions()
        # 
        #time_end = time.time()
        #
        #elapsed_time = time_end - time_start
        #print(f"Function 'getPossibleActions' took {elapsed_time:.6f} seconds to run.")

        for action in actions:
            action = (action[0],action[1])
            # import pdb
            # pdb.set_trace()
            if action not in node.children.keys():
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        #raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = (
                    (child.totalReward / child.numVisits) +  # 效用项
                    explorationValue * math.sqrt(2 * math.log(node.numVisits) / child.numVisits) -  # 探索项
                    self.epsilon * (child.state.removed_weight / (child.state.remaining_budget + 1e-3))  # 成本项
            )

            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
