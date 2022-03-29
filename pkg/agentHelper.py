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
from helpPlan import HelpPlan
from helpPlan import Node

##Importa o Planner
sys.path.append('pkg/planner')
from planner import Planner

## Classe que define o Agente
class AgentSearcher:
    def __init__(self, model, batteryHelper, timeHelper, packageHelper, map, victimsPos):
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
        self.map = map
        self.victimsQueue = victimsPos

        """
        DEFINE OS PLANOS DE EXECUÇÃO DO AGENTE
        """

        ## Custo da solução
        self.costAll = 0

        ## adicionar crencas sobre o estado do ambiente ao plano - neste exemplo, o agente faz uma copia do que existe no ambiente.
        ## Em situacoes de exploracao, o agente deve aprender em tempo de execucao onde estao as paredes
        self.plan = HelpPlan(self.model.rows, self.model.columns, self.prob.initialState, self.currentState, self.map, "goal", self.mesh )
        self.plan.setWalls(model.maze.walls)

        ## Adiciona o(s) planos a biblioteca de planos do agente
        self.libPlan = []
        self.libPlan.append(self.plan)

        ## inicializa acao do ciclo anterior com o estado esperado
        self.previousAction = "nop"    ## nenhuma (no operation)
        self.expectedState = self.currentState

        ## inicializa a bateria e tempo do agente vasculhador
        self.battery = batteryHelper
        self.maxBaterry = batteryHelper
        self.time = timeHelper
        self.packages = packageHelper

    ## Metodo que define a deliberacao do agente
    def deliberate(self):

        if victimsQueue.empty():
            print("Nenhuma vítima encontrada pelo agente vasculhador. Encerrando execucao: 0 vitimas salvas")
            return -1

        while not victimsQueue.empty():

            victimPos = self.victimsQueue.pop()
            self.plan.updateGoal(victimsPos)
            self.plan.updateInitialState(self.positionSensor())
            victimPath = self.plan.findPath()

            self.plan.updateGoal(self.prob.initialState)
            self.plan.updateInitialState(self.positionSensor())
            homePath = self.plan.findPath()

            if(2*victimPath[victimPos][1] + homePath[self.prob.initialState][1] > self.battery or
            2*victimPath[victimPos][1] + homePath[self.prob.initialState][1] > self.time ):
                self.plan.makePath(homePath)
                self.victimsQueue.append(victimPos)
                self.walkPath(self.prob.initialState)

            else:
                self.plan.makePath(victimPath)
                self.packages -= 1
                self.walkPath(victimsPos)

    def walkPath(self, goal):
        while self.positionSensor() not goal:
            result = self.plan.chooseAction()
            self.executeGo(result[0][0])
            self.battery -= result[1]
            self.time -= result[2]

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
