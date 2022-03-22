
from state import State

class ReturnPlan:
    def __init__(self, maxRows, maxColumns, goal, initialState, name = "none", mesh = "square", map):
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
        # TODO!

    def turnMatrixToGraph(self):
        # TODO!

    def findPath(self):
        # TODO!

    def calculateHeuristic(self, row, col):
        return sqrt((row - self.goalPos.row)*(row - self.goalPos.row) + (col - self.goalPos.col)*(col - self.goalPos.col))

    def do(self):
        """
        Método utilizado para o polimorfismo dos planos

        Retorna o movimento e o estado do plano (False = nao concluido, True = Concluido)
        """

        nextMove = self.move()
        return (nextMove[1], self.goalPos == State(nextMove[0][0], nextMove[0][1]))
