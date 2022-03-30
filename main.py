import sys
import os
import time

## Importa as classes que serao usadas
sys.path.append('pkg')
from model import Model
from agentRnd import AgentRnd
from agentSearcher import AgentSearcher
from agentHelper import AgentHelper


## Metodo utilizado para permitir que o usuario construa o labirindo clicando em cima
def buildMaze(model):
    model.drawToBuild()
    step = model.getStep()
    while step == "build":
        model.drawToBuild()
        step = model.getStep()
    ## Atualiza o labirinto
    model.updateMaze()

def main():
    # Lê arquivo config.txt
    arq = open(os.path.join("config_data","config.txt"),"r")
    configDict = {}
    for line in arq:
        ## O formato de cada linha é:var=valor
        ## As variáveis são
        ##  maxLin, maxCol que definem o tamanho do labirinto
        ## Tv e Ts: tempo limite para vasculhar e tempo para salvar
        ## Bv e Bs: bateria inicial disponível ao agente vasculhador e ao socorrista
        ## Ks :capacidade de carregar suprimentos em número de pacotes (somente para o ag. socorrista)

        values = line.split("=")
        configDict[values[0]] = int(values[1])

    print("dicionario config: ", configDict)

    # Cria o ambiente (modelo) = Labirinto com suas paredes
    mesh = "square"

    ## nome do arquivo de configuracao do ambiente - deve estar na pasta <proj>/config_data
    loadMaze = "ambiente"

    model = Model(configDict["maxLin"], configDict["maxCol"], mesh, loadMaze)
    buildMaze(model)

    #model.maze.board.posAgent
    #model.maze.board.posGoal
    # Define a posição inicial do agente no ambiente - corresponde ao estado inicial
    model.setagentSearcherPos(configDict["maxLin"]-1, 0)
    model.setagentHelperPos(configDict["maxLin"]-1, 0)
    #model.setGoalPos(model.maze.board.posGoal[0],model.maze.board.posGoal[1])
    model.draw()

    # Cria um agente vasculhador
    agentSearcher = AgentSearcher(model, configDict["Bv"], configDict["Tv"])
    ## Ciclo de raciocínio do agente vasculhador
    agentSearcher.deliberate()
    while agentSearcher.deliberate() != -1:
        model.draw()
        time.sleep(0.005) # para dar tempo de visualizar as movimentacoes do agente no labirinto
    model.draw()

    # victimsQueue = agentSearcher.getVictimsQueue()
    # while victimsQueue:
    #     victimPos = victimsQueue.pop()
    #     print(victimPos.row, victimPos.col)

    agentHelper = AgentHelper(model, configDict["Bs"], configDict["Ts"], configDict["Ks"], agentSearcher.getMap(), agentSearcher.getVictimsQueue())

    model.draw()
    agentHelper.deliberate()
    while agentHelper.moves:
        agentHelper.executeGo(agentHelper.moves.pop(0))
        model.draw()
        time.sleep(0.1)
    model.draw()

if __name__ == '__main__':
    main()
