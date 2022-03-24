from queue import PriorityQueue
from state import State

class Node:
    def __init__(self, position):
        self.position = position
        self.neighbors = []

    def updateNeighbors(self, map):
        self.neighbors = []
        if self.position.row < len(map) - 1 and map[self.position.row + 1][self.position.col] == 1:
            self.neighbors.append(State(self.position.row + 1, self.position.col)) #SOUTH

        if self.position.row > 0 and map[self.position.row - 1][self.position.col] == 1:
            self.neighbors.append(State(self.position.row - 1, self.position.col)) #NORTH

        if self.position.col < len(map[0]) - 1 and map[self.position.row][self.position.col + 1] == 1:
            self.neighbors.append(State(self.position.row, self.position.col + 1)) #EAST

        if self.position.row > 0 and map[self.position.row][self.position.col - 1] == 1:
            self.neighbors.append(State(self.position.row, self.position.col - 1)) #WEST

        if self.position.row < len(map) - 1 and self.position.col < len(map[0]) - 1 and map[self.position.row + 1][self.position.col + 1] == 1:
            self.neighbors.append(State(self.position.row + 1, self.position.col + 1)) #SOUTHEAST

        if self.position.row > 0 and self.position.col < len(map[0]) - 1 and map[self.position.row - 1][self.position.col + 1] == 1:
            self.neighbors.append(State(self.position.row - 1, self.position.col + 1)) #NORTHEAST

        if self.position.row < len(map) - 1 and self.position.col > 0 and map[self.position.row + 1][self.position.col - 1] == 1:
            self.neighbors.append(State(self.position.row + 1, self.position.col - 1)) #SOUTHWEST

        if self.position.row > 0 and self.position.col > 0 and map[self.position.row - 1][self.position.col - 1] == 1:
            self.neighbors.append(State(self.position.row - 1, self.position.col - 1)) #NORTHWEST

class ReturnPlan:
    def __init__(self, maxRows, maxColumns, goal, initialState, map, name = "none", mesh = "square"):
        """
        Define as variaveis necessárias para a utilização do search plan por um unico agente.
        """
        self.walls = []
        self.maxRows = maxRows
        self.maxColumns = maxColumns
        self.initialState = initialState
        self.currentState = initialState
        self.goalPos = goal
        self.actions = []
        self.map = map

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


    def updateCurrentState(self, state):
         self.currentState = state

    def updateMap(self, map):
        self.map = map

    def getNeighbors(self):
        pass

    def matrixToGraph(self):
        pass

    def aStarSearch(self):

        nodeDict = {}
        for i in range(self.maxRows):
            for j in range(self.maxColumns):
                if(self.map[i][j] == 1):
                    node = Node(State(i,j))
                    node.updateNeighbors(self.map)
                    key = str(i) + ';' + str(j)
                    nodeDict.update({key: node})

        count = 0
        openList = PriorityQueue()
        key = str(self.initialState.row) + ';' + str(self.initialState.col)
        openList.put((0, count, nodeDict[key]))
        cameFrom = {}
        gValue = {spot: float("inf") for i in range(self.maxRows) for j in range(self.maxColumns)}
        gValue[self.initialState] = 0
        fValue = {spot: float("inf") for i in range(self.maxRows) for j in range(self.maxColumns)}
        fValue[self.initialState] = calculateHeuristic(self.initialState.row, self.initialState.col)

        openListHash = {self.initialState}

        while not openList.empty():
            current = openList.get()[2]
            openListHash.remove(current.position)

            if current == goalPos:
                return True

            #for neighbors in current.neighbors:


    def calculateHeuristic(self, row, col):
        return sqrt((row - self.goalPos.row)*(row - self.goalPos.row) + (col - self.goalPos.col)*(col - self.goalPos.col))

    def do(self):
        """
        Método utilizado para o polimorfismo dos planos

        Retorna o movimento e o estado do plano (False = nao concluido, True = Concluido)
        """

        nextMove = self.move()
        return (nextMove[1], self.goalPos == State(nextMove[0][0], nextMove[0][1]))
