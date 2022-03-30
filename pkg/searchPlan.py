
from state import State

class SearchPlan:
    def __init__(self, maxRows, maxColumns, goal, initialState, name = "none", mesh = "square"):
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

        self.untried = []
        for i in range(maxRows):
            self.untried.append([])
            for j in range(maxColumns):
                self.untried[i].append(["N", "S", "L", "O", "NE", "SO", "NO", "SE"])


        self.unbacktracked = []
        for i in range(maxRows):
            self.unbacktracked.append([])
            for j in range(maxColumns):
                self.unbacktracked[i].append([])

        self.map = []
        self.compass = { "N" : (0, "S", 1),
                         "S" : (1, "N", 0),
                         "L" : (2, "O", 3),
                         "O" : (3, "L", 2),
                         "NE" : (4, "SO", 5),
                         "SO" : (5, "NE", 4),
                         "NO" : (6, "SE", 7),
                         "SE" : (7, "NO", 6)
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
         self.currentState = state

    def isPossibleToMove(self, toState):
        """Verifica se eh possivel ir da posicao atual para o estado (lin, col) considerando
        a posicao das paredes do labirinto e movimentos na diagonal
        @param toState: instancia da classe State - um par (lin, col) - que aqui indica a posicao futura
        @return: True quando é possivel ir do estado atual para o estado futuro """


        ## vai para fora do labirinto
        if (toState.col < 0 or toState.row < 0):
            return False

        if (toState.col >= self.maxColumns or toState.row >= self.maxRows):
            return False

        if len(self.walls) == 0:
            return True

        ## vai para cima de uma parede
        if (toState.row, toState.col) in self.walls:
            return False

        # vai na diagonal? Caso sim, nao pode ter paredes acima & dir. ou acima & esq. ou abaixo & dir. ou abaixo & esq.
        delta_row = toState.row - self.currentState.row
        delta_col = toState.col - self.currentState.col

        ## o movimento eh na diagonal
        if (delta_row !=0 and delta_col != 0):
            if (self.currentState.row + delta_row, self.currentState.col) in self.walls and (self.currentState.row, self.currentState.col + delta_col) in self.walls:
                return False

        return True

    def setNextPosition(self):

        if self.untried[self.currentState.row][self.currentState.col]:
            movDirection = self.untried[self.currentState.row][self.currentState.col].pop(0)
            movePos = { "N" : (-1, 0),
                    "S" : (1, 0),
                    "L" : (0, 1),
                    "O" : (0, -1),
                    "NE" : (-1, 1),
                    "SO" : (1, -1),
                    "NO" : (-1, -1),
                    "SE" : (1, 1)
                    }

        elif self.unbacktracked[self.currentState.row][self.currentState.col]:
            movDirection = self.unbacktracked[self.currentState.row][self.currentState.col].pop(0)

        state = State(self.currentState.row + movePos[movDirection][0], self.currentState.col + movePos[movDirection][1])

        hitWall = 0
        if not self.isPossibleToMove(state):
            state = self.currentState
            hitWall = 1
        return movDirection, state, hitWall


    def chooseAction(self):
        """ Escolhe o proximo movimento de forma aleatoria.
        Eh a acao que vai ser executada pelo agente.
        @return: tupla contendo a acao (direcao) e uma instância da classe State que representa a posição esperada após a execução
        """

        ## Tenta encontrar um movimento possivel dentro do tabuleiro
        result = self.setNextPosition()

        self.map[result[1].row][result[1].col] = 1
        # self.map[self.currentState.row][self.currentState.row][self.compass[result[0]][0]] = [result[1].row, result[1].col]
        # self.map[result[1].row][result[1].col][self.compass[result[0]][2]] = [self.currentState.row, self.currentState.col]

        if self.compass[result[0]][1] in self.untried[result[1].row][result[1].col] and not result[2]:
            self.untried[result[1].row][result[1].col].remove(self.compass[result[0]][1])
            self.unbacktracked[result[1].row][result[1].col].append(self.compass[result[0]][1])

        if(self.compass[result[0]][0] < 4):
            batteryCost = 1
            timeCost = 1
        else:
            batteryCost = 1.5
            timeCost = 1.5

        return result, batteryCost, timeCost, self.map


    def do(self):
        """
        Método utilizado para o polimorfismo dos planos

        Retorna o movimento e o estado do plano (False = nao concluido, True = Concluido)
        """

        nextMove = self.move()
        return (nextMove[1], self.goalPos == State(nextMove[0][0], nextMove[0][1]))
