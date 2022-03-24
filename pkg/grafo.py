class Graph:

    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[] for i in range(self.vertices)]

    def adiciona_aresta(self, u, v, peso):
        # estamos pensando em grafo direcionado com peso nas arestas
        self.graph[u-1].append([v, peso])

        # self.grafo[v-1].append([u,peso]) se o grafo não for direcionado

    def turnMatrixToGraph(self, map):
        for i in range(len(map)):
            for j in range(len(map[0])):
                if((i-1)>0 & (j-1>0) & map[i-1][j-1]==1):
                    self.adiciona_aresta()

                if((i-1)>0 & map[i-1][j]==1):
                    self.adiciona_aresta()

                if((i-1)>0 & (j+1<len(map)) & map[i-1][j+1]==1):
                    self.adiciona_aresta()

                if((j-1>0) & map[i][j-1]==1):
                    self.adiciona_aresta()

                if((j+1<len(map)) & map[i][j+1]==1):
                    self.adiciona_aresta()

                if((i+1)<len(map[0]) & (j-1>0) & map[i+1][j-1]==1):
                    self.adiciona_aresta()

                if((i+1)<len(map[0]) & map[i+1][j]==1):
                    self.adiciona_aresta()

                if((i+1)<len(map[0]) & (j+1<len(map)) & map[i+1][j+1]==1):
                    self.adiciona_aresta()

    # def calculaVolta(self, map, posAtualX, posAtualY, posVoltaX, posVoltaY, linhas, colunas):

    #     #Copia matriz de camihos conhecidos, onde lugares conhecidos são dados por -1
    #     #Objetivo deve ser dado por -2(falta programar)
    #     matrizVolta = []
    #     for i in range(linhas):
    #         for j in range(colunas):
    #             matrizVolta[i][j]=-map[i][j]

    #     flag=0
    #     while(flag==0):


        
        
                



    # def mostra_lista(self):
    #     for i in range(self.vertices):
    #         print(f'{i+1}:', end='  ')
    #         for j in self.grafo[i]:
    #             print(f'{j}  ->', end='  ')
    #         print('')

