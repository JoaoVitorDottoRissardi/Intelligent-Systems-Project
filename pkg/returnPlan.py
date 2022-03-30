from queue import PriorityQueue
from state import State
from math import sqrt

class Node:
    def __init__(self, position):
        self.position = position
        self.neighbors = []

    def updateNeighbors(self, map):
        self.neighbors = []
        if self.position.row < len(map) - 1 and map[self.position.row + 1][self.position.col] == 1:
            self.neighbors.append((State(self.position.row + 1, self.position.col), 1)) #SOUTH

        if self.position.row > 0 and map[self.position.row - 1][self.position.col] == 1:
            self.neighbors.append((State(self.position.row - 1, self.position.col), 1)) #NORTH

        if self.position.col < len(map[0]) - 1 and map[self.position.row][self.position.col + 1] == 1:
            self.neighbors.append((State(self.position.row, self.position.col + 1), 1)) #EAST

        if self.position.col > 0 and map[self.position.row][self.position.col - 1] == 1:
            self.neighbors.append((State(self.position.row, self.position.col - 1), 1)) #WEST

        if self.position.row < len(map) - 1 and self.position.col < len(map[0]) - 1 and map[self.position.row + 1][self.position.col + 1] == 1:
            self.neighbors.append((State(self.position.row + 1, self.position.col + 1), 1.5)) #SOUTHEAST

        if self.position.row > 0 and self.position.col < len(map[0]) - 1 and map[self.position.row - 1][self.position.col + 1] == 1:
            self.neighbors.append((State(self.position.row - 1, self.position.col + 1), 1.5)) #NORTHEAST

        if self.position.row < len(map) - 1 and self.position.col > 0 and map[self.position.row + 1][self.position.col - 1] == 1:
            self.neighbors.append((State(self.position.row + 1, self.position.col - 1), 1.5)) #SOUTHWEST

        if self.position.row > 0 and self.position.col > 0 and map[self.position.row - 1][self.position.col - 1] == 1:
            self.neighbors.append((State(self.position.row - 1, self.position.col - 1), 1.5)) #NORTHWEST

class ReturnPlan:
    def __init__(self, maxRows, maxColumns, goal, initialState, map, name = "none", mesh = "square"):
        """
        Define as variaveis necessárias para a utilização do return plan por um unico agente.
        """
        self.walls = []
        self.maxRows = maxRows
        self.maxColumns = maxColumns
        self.initialState = initialState
        self.goalPos = goal
        self.map = map
        self.returnPath = []
        self.compass = { "N" : (-1,0),
                         "S" : (1,0),
                         "L" : (0,1),
                         "O" : (0,-1),
                         "NE" : (-1,1),
                         "SO" : (1,-1),
                         "NO" : (-1,-1),
                         "SE" : (1,1)
                        }

    def setWalls(self, walls):
        row = 0
        col = 0
        for i in walls:
            col = 0
            for j in i:
                if j == 1:
                    self.walls.append((row, col))
                col += 1
            row += 1

    def updateMap(self, map):
        self.map = map

    def updateCurrentState(self, state):
        self.initialState = state

    def calculateHeuristic(self, row, col):
        return sqrt((row - self.goalPos.row)*(row - self.goalPos.row) + (col - self.goalPos.col)*(col - self.goalPos.col))

    def findPath(self):
        cameFrom = {}
        if self.initialState == self.goalPos:
            cameFrom[self.initialState] = (Node(self.goalPos), 0)
            return cameFrom

        nodeDict = {}
        for i in range(self.maxRows):
            for j in range(self.maxColumns):
                if(self.map[i][j] == 1):
                    node = Node(State(i,j))
                    node.updateNeighbors(self.map)
                    nodeDict.update({node.position: node})

        count = 0
        openList = PriorityQueue()
        openList.put((0, count, nodeDict[self.initialState]))
        gValue = {State(i,j): float("inf") for i in range(self.maxRows) for j in range(self.maxColumns)}
        gValue[self.initialState] = 0
        fValue = {State(i,j): float("inf") for i in range(self.maxRows) for j in range(self.maxColumns)}
        fValue[self.initialState] = self.calculateHeuristic(self.initialState.row, self.initialState.col)
        openListHash = {self.initialState}

        while not openList.empty():
            current = openList.get()[2]
            openListHash.remove(current.position)

            if current.position == self.goalPos:
                return cameFrom
            #print("Current:", current.position)
            for neighbor in current.neighbors:
                tempGValue = gValue[current.position] + neighbor[1]
                #print("Neighbors:", neighbor[0])
                if tempGValue < gValue[neighbor[0]]:
                    cameFrom[neighbor[0]] = (current, tempGValue)
                    gValue[neighbor[0]] = tempGValue
                    fValue[neighbor[0]] = tempGValue + self.calculateHeuristic(neighbor[0].row, neighbor[0].col)
                    if neighbor[0] not in openListHash:
                        count += 1
                        openList.put((fValue[neighbor[0]], count, nodeDict[neighbor[0]]))
                        openListHash.add(neighbor[0])

    def makePath(self, cameFrom, start, current):
        moveStack = []
        movePos = {(-1, 0) : "N",
                (1, 0) : "S",
                (0, 1): "L",
                (0, -1) : "O",
                (-1, 1) : "NE",
                (1, -1) : "SO",
                (-1, -1) : "NO",
                (1, 1) : "SE"
                }
        while current.position != start:
            direction = (current.position.row - cameFrom[current.position][0].position.row, current.position.col - cameFrom[current.position][0].position.col)
            moveStack.append(movePos[direction])
            current = cameFrom[current.position][0]

        self.returnPath = moveStack


    def chooseAction(self):
        nextMove = self.returnPath.pop()
        result = (nextMove, State(self.initialState.row + self.compass[nextMove][0], self.initialState.col + self.compass[nextMove][1]))
        print(result[1])
        batteryCost = 0
        timeCost = 0
        if(result[0] == "N" or result[0] == "S" or result[0] == "L" or result[0] == "O"):
            batteryCost += 1
            timeCost += 1
        else:
            batteryCost += 1.5
            timeCost += 1.5

        return result, batteryCost, timeCost, self.map

    def do(self):
        """
        Método utilizado para o polimorfismo dos planos

        Retorna o movimento e o estado do plano (False = nao concluido, True = Concluido)
        """

        nextMove = self.move()
        return (nextMove[1], self.goalPos == State(nextMove[0][0], nextMove[0][1]))
