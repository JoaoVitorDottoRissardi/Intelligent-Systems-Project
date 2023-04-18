from collections import deque

class Graph:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]
    
    def heuristic(self, n, goal):
        return ((n[0] - goal[0])**2 + (n[1] - goal[1])**2)
        

    def a_star_search(self, start_node, stop_node):
        
        open_list = set([start_node])
        closed_list = set([])

        g = {}
        g[start_node] = 0

        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:

            n = None

            for v in open_list:
                if n == None or g[v] + self.heuristic(v, stop_node) < g[n] + self.heuristic(n, stop_node):
                    n = v

            if n == None:
                print("Path does not exist")
                return None

            if n == stop_node:
                reconst_path = [stop_node]

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]
                
                reconst_path.append(start_node)

                reconst_path.reverse()

                #print('Path found: {}'.format(reconst_path))
                return reconst_path

            for (m, weight) in self.get_neighbors(n):

                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            open_list.remove(n)
            closed_list.add(n)

        print("Path does not exist")
        return None