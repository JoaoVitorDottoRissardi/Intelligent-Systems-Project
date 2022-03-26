## AGENTE RANDOM
### @Author: Luan Klein e Tacla (UTFPR)
### Agente que fixa um objetivo aleatório e anda aleatoriamente pelo labirinto até encontrá-lo.
### Executa raciocíni on-line: percebe --> [delibera] --> executa ação --> percebe --> ...
import sys
import numpy as np
## Importa Classes necessarias para o funcionamento
from model import Model
from problem import Problem
from state import State
from random import randint

## Importa o algoritmo para o plano
from randomPlan import RandomPlan
from searchPlan import SearchPlan
from returnPlan import ReturnPlan
from returnPlan import Node

##Importa o Planner
sys.path.append('pkg/planner')
from planner import Planner

## Classe que define o Agente
class AgentSearcher:
    def __init__(self, model, batterySearcher, timeSearcher):
        """
        Construtor do agente random
        @param model referencia o ambiente onde o agente estah situado
        """

        self.model = model

        ## Pega o tipo de mesh, que está no model (influência na movimentação)
        self.mesh = self.model.mesh


        ## Cria a instância do problema na mente do agente (sao suas crencas)
        self.prob = Problem()
        self.prob.createMaze(model.rows, model.columns, model.maze)


        # O agente le sua posica no ambiente por meio do sensor
        initial = self.positionSensor()
        self.prob.defInitialState(initial.row, initial.col)
        print("*** Estado inicial do agente: ", self.prob.initialState)

        # Define o estado atual do agente = estado inicial
        self.currentState = self.prob.initialState

        # Define o estado objetivo:
        # definimos um estado objetivo aleatorio
        # self.prob.defGoalState(randint(0,model.rows-1), randint(0,model.columns-1))

        # definimos um estado objetivo que veio do arquivo ambiente.txt
        self.prob.defGoalState(model.maze.board.posGoal[0],model.maze.board.posGoal[1])
        print("*** Objetivo do agente: ", self.prob.goalState)
        print("*** Total de vitimas existentes no ambiente: ", self.model.getNumberOfVictims())

        #define o mapa que o agente carregará consigo
        self.map = []
        for i in range(self.model.rows):
            self.map.append([])
            for j in range(self.model.columns):
                self.map[i].append(0)
        self.map[self.prob.initialState.row][self.prob.initialState.col] = 1

        """
        DEFINE OS PLANOS DE EXECUÇÃO DO AGENTE
        """

        ## Custo da solução
        self.costAll = 0

        ## Cria a instancia do plano para se movimentar aleatoriamente no labirinto (sem nenhuma acao)
        self.plan = SearchPlan(model.rows, model.columns, self.prob.goalState, initial, "goal", self.mesh)

        ## adicionar crencas sobre o estado do ambiente ao plano - neste exemplo, o agente faz uma copia do que existe no ambiente.
        ## Em situacoes de exploracao, o agente deve aprender em tempo de execucao onde estao as paredes
        self.plan.setWalls(model.maze.walls)

        ## Adiciona o(s) planos a biblioteca de planos do agente
        self.libPlan = []
        self.libPlan.append(self.plan)
        self.plan = ReturnPlan(self.model.rows, self.model.columns, self.prob.initialState, self.currentState, self.map, "goal", self.mesh )
        self.libPlan.append(self.plan)

        ## inicializa acao do ciclo anterior com o estado esperado
        self.previousAction = "nop"    ## nenhuma (no operation)
        self.expectedState = self.currentState

        ## inicializa a bateria e tempo do agente vasculhador
        self.battery = batterySearcher
        self.maxBaterry = batterySearcher
        self.time = timeSearcher

    ## Metodo que define a deliberacao do agente
    def deliberate(self):
        ## Verifica se há algum plano a ser executado
        if len(self.libPlan) == 0:
            return -1   ## fim da execucao do agente, acabaram os planos

        print("\n*** Inicio do ciclo raciocinio ***")
        print("Pos agente no amb.: ", self.positionSensor())

        ## Redefine o estado atual do agente de acordo com o resultado da execução da ação do ciclo anterior
        self.currentState = self.positionSensor()
        self.libPlan[0].updateCurrentState(self.currentState) # atualiza o current state no plano
        self.libPlan[0].updateMap(self.map) # atualiza o mapa no plano
        self.libPlan[1].updateCurrentState(self.currentState) # atualiza o current state no plano
        self.libPlan[1].updateMap(self.map) # atualiza o mapa no plano
        self.plan = self.libPlan[0]
        print("Ag cre que esta em: ", self.currentState)

        ## Verifica se a execução do acao do ciclo anterior funcionou ou nao
        if not (self.currentState == self.expectedState):
            print("---> erro na execucao da acao ", self.previousAction, ": esperava estar em ", self.expectedState, ", mas estou em ", self.currentState)

        # ## Funcionou ou nao, vou somar o custo da acao com o total
        # self.costAll += self.prob.getActionCost(self.previousAction)
        # print ("Custo até o momento (com a ação escolhida):", self.costAll)

        ## Verifica se atingiu o estado objetivo
        ## Poderia ser outra condição, como atingiu o custo máximo de operação
        # if self.prob.goalTest(self.currentState):
        #     print("!!! Objetivo atingido !!!")
        #     del self.libPlan[0]  ## retira plano da biblioteca

        ## Verifica se tem vitima na posicao atual
        victimId = self.victimPresenceSensor()
        if victimId > 0:
            print ("vitima encontrada em ", self.currentState, " id: ", victimId, " sinais vitais: ", self.victimVitalSignalsSensor(victimId))
            print ("vitima encontrada em ", self.currentState, " id: ", victimId, " dif de acesso: ", self.victimDiffOfAcessSensor(victimId))
            
            self.battery -= 2
            self.time -= 2

        # Calcula o melhor caminho para retornar e avalia se precisa executá-lo ao não
        goalNode = Node(self.prob.initialState)
        returnPath = self.libPlan[1].findPath()
        print("Bateria do Agente: " + str(self.battery))
        print("Tempo restante do Agente: " + str(self.time))
        if returnPath:
            print("Custo calculado para voltar para base: " + str(returnPath[goalNode.position][1]))
        else:
            print("Custo calculado para voltar para base: 0 ")
        if(returnPath and (returnPath[goalNode.position][1] >= (self.battery - 4.5) or returnPath[goalNode.position][1] >= (self.time - 4.5))):
            self.plan = self.libPlan[1]
            self.plan.makePath(returnPath, self.currentState, goalNode)
            print(self.plan.returnPath)
        ## Define a proxima acao a ser executada
        ## currentAction eh uma tupla na forma: <direcao>, <state>
        result = self.plan.chooseAction()

        # Caso a ação escolhida vá deixar o agente sem tempo para voltar ele fica parado
        if (self.time - (2*result[2])) <= 0:
            print("Ag deliberou pela acao: ", "STAY", " o estado resultado esperado é a posição atual")
            self.executeGo("ST")
        ## Executa esse acao, atraves do metodo executeGo
        else:
            print("Ag deliberou pela acao: ", result[0][0], " o estado resultado esperado é: ", result[0][1])
            self.executeGo(result[0][0])
        self.previousAction = result[0][0]
        self.expectedState = result[0][1]

        self.battery -= result[1]
        self.time -= result[2]
        self.map = result[3]

        # A bateria do agente acabou e ele estava fora da base
        if (self.battery <= 0  and self.currentState != self.prob.initialState ):
            print("Agente sem bateria ; Agente fora da base ; Encerrando: 0 vitimas salvas")
            del self.libPlan[0]
            del self.libPlan[0]
            return 1

        # A bateria do agente acabou e ele estava dentro da base
        if self.battery <=0 :
            print("Agente recarregando")
            self.baterry = self.maxBaterry
            self.time -= 240

        # O tempo do agente acabou e ele estava fora da base
        if self.time <= 0  and self.currentState != self.prob.initialState:
            print("Tempo expirado ; Agente fora da base ; Encerrando: 0 vitimas salvas")
            del self.libPlan[0]
            del self.libPlan[0]
            return 1
            # for i in range(self.model.rows):
            #     print(self.map[i])

        # O tempo do agente acabou e ele estava dentro da base
        if self.time <= 0:
            print("Tempo expirado ; Passando as informacoes para o agente socorrista")
            del self.libPlan[0]
            del self.libPlan[0]
            return 1

    ## Metodo que executa as acoes
    def executeGo(self, action):
        """Atuador: solicita ao agente físico para executar a acao.
        @param direction: Direcao da acao do agente {"N", "S", ...}
        @return 1 caso movimentacao tenha sido executada corretamente """

        ## Passa a acao para o modelo
        result = self.model.go(action)

        ## Se o resultado for True, significa que a acao foi completada com sucesso, e ja pode ser removida do plano
        ## if (result[1]): ## atingiu objetivo ## TACLA 20220311
        ##    del self.plan[0]
        ##    self.actionDo((2,1), True)


    ## Metodo que pega a posicao real do agente no ambiente
    def positionSensor(self):
        """Simula um sensor que realiza a leitura do posição atual no ambiente.
        @return instancia da classe Estado que representa a posição atual do agente no labirinto."""
        pos = self.model.agentPos
        return State(pos[0],pos[1])

    def victimPresenceSensor(self):
        """Simula um sensor que realiza a deteccao de presenca de vitima na posicao onde o agente se encontra no ambiente
           @return retorna o id da vítima"""
        return self.model.isThereVictim()

    def victimVitalSignalsSensor(self, victimId):
        """Simula um sensor que realiza a leitura dos sinais da vitima
        @param o id da vítima
        @return a lista de sinais vitais (ou uma lista vazia se não tem vítima com o id)"""
        return self.model.getVictimVitalSignals(victimId)

    def victimDiffOfAcessSensor(self, victimId):
        """Simula um sensor que realiza a leitura dos dados relativos à dificuldade de acesso a vítima
        @param o id da vítima
        @return a lista dos dados de dificuldade (ou uma lista vazia se não tem vítima com o id)"""
        return self.model.getDifficultyOfAcess(victimId)

    ## Metodo que atualiza a biblioteca de planos, de acordo com o estado atual do agente
    def updateLibPlan(self):
        for i in self.libPlan:
            i.updateCurrentState(self.currentState)

    def actionDo(self, posAction, action = True):
        self.model.do(posAction, action)
