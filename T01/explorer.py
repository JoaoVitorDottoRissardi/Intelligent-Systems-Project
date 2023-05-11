## EXPLORER AGENT
### @Author: Tacla, UTFPR
### It walks randomly in the environment looking for victims.

from hashlib import new
from lib2to3.pgen2.token import EQUAL
from multiprocessing.connection import wait
import sys
import os
import random
import time
from graph import Graph
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent
from abc import ABC, abstractmethod


class Explorer(AbstractAgent):
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env referencia o ambiente
        @config_file: the absolute path to the explorer's config file
        @param resc referencia o rescuer para poder acorda-lo
        """

        super().__init__(env, config_file)
        
        # Specific initialization for the rescuer
        self.resc = resc           # reference to the rescuer agent
        self.rtime = self.TLIM     # remaining time to explore
        self.time_to_return = 0    # time needed to return to base
        self.return_flag = 0       # flag to indicate agent is returning
        self.path_to_base = []     # path to return home 
        self.victims_list = []     # list of victims positions

        self.current_position = (0,0) 

        self.map = {
            (0,0) : []
        }

        self.moves = {
            (0,0) : [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (-1, -1), (1, 1)]
        }

        self.backtracking_list = []    


    def deliberate(self) -> bool:

        if self.return_flag:

            #print("Returning home")

            #print("Current position:", self.current_position)
            #print("Path to base:", self.path_to_base)
            #print("Cost to return:", self.time_to_return)

            if self.current_position == (0,0):
                #print("Exploration finished, sending recuer agent")    
                self.resc.go_save_victims(self.map,self.victims_list)
                return False

            else:
                (dx, dy) = (self.path_to_base[0][0] -  self.current_position[0], self.path_to_base[0][1] - self.current_position[1])
                self.current_position = self.path_to_base.pop(0)
                
            
        else:

            #Calculate cost and path to return
            graph = Graph(self.map)
            self.path_to_base = graph.a_star_search(self.current_position, (0,0))
            self.path_to_base.pop()
            self.path_to_base.pop(0)
            self.time_to_return = self.calculate_cost_to_return()

            # Returns home if time is over
            if self.rtime < self.time_to_return + 5: 
                #print("Need to return!")
                self.return_flag = 1
                return True


            backtracking_flag = 0
            
            # Backtrack if there is no more moves
            if(len(self.moves[self.current_position]) == 0):
                #print("Backtracking!")
                if len(self.backtracking_list) == 0:
                    self.resc.go_save_victims(self.map,self.victims_list)
                    return False

                (dx, dy) = self.backtracking_list.pop()
                new_position = (self.current_position[0] + dx, self.current_position[1] + dy)
                backtracking_flag = 1

            else:
            # Decide which is going to be the next move
                (dx, dy) = self.moves[self.current_position][0]

                self.moves[self.current_position].pop(0)

                new_position = (self.current_position[0] + dx, self.current_position[1] + dy)

                # If agent knows it is going to hit a wall or repeat position aborts movement e tries again
                if new_position in self.moves:
                    return True
        
        # Moves the body to another position
        result = self.body.walk(dx, dy)

        # Update remaining time
        if dx != 0 and dy != 0:
            self.rtime -= self.COST_DIAG
        else:
            self.rtime -= self.COST_LINE

        # Test the result of the walk action
        if result == PhysAgent.BUMPED:
            #print("Hit Wall:", new_position)
            self.moves.update({new_position : (0,0)})

        if result == PhysAgent.EXECUTED:
            if not self.return_flag:
                seq = self.body.check_for_victim()
                if seq >= 0 and new_position not in self.moves:
                    vs = self.body.read_vital_signals(seq)
                    #print("Victim position:", new_position)
                    #print("Victim vital signs:", vs)
                    self.victims_list.append((new_position, vs))
                    self.rtime -= self.COST_READ

                if new_position not in self.moves:
                    self.moves.update({new_position : [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (-1, -1), (1, 1)]})

                self.current_position = new_position

                if not backtracking_flag:

                    self.backtracking_list.append((dx*-1, dy*-1))

                    self.update_map(new_position)

                #print(self.map)
            

        #print("Energy:", self.rtime)
 
        return True

    def calculate_cost_to_return(self):

        coordinate_1 = self.current_position

        total_cost = 0

        for i in range(0, len(self.path_to_base)):

            coordinate_2 = self.path_to_base[i]
            (dx, dy) = (coordinate_2[0] - coordinate_1[0], coordinate_2[1] - coordinate_1[1])

            if dx == 0 or dy == 0:
                total_cost += self.COST_LINE
            else:
                total_cost += self.COST_DIAG

            coordinate_1 = coordinate_2

        return total_cost 

    def update_map(self, coordinate):

        #print("Updating for:", coordinate)

        self.map.update({coordinate : []})

        for i in range(-1, 2):

            for j in range (-1, 2):

                if i != 0 or j != 0:

                    neighbor = (coordinate[0] + i, coordinate[1] + j)

                    if neighbor in self.moves:

                        if self.moves[neighbor] != (0,0):

                            if i == 0 or j == 0 :
                                temp_list = self.map[neighbor]
                                temp_list.append((coordinate, self.COST_LINE))
                                self.map.update({neighbor : temp_list})
                                
                                temp_list = self.map[coordinate]
                                temp_list.append((neighbor, self.COST_DIAG))
                                self.map.update({coordinate : temp_list})
                                

                            else:
                                temp_list = self.map[neighbor]
                                temp_list.append((coordinate, self.COST_DIAG))
                                self.map.update({neighbor : temp_list})

                                temp_list = self.map[coordinate]
                                temp_list.append((neighbor, self.COST_DIAG))
                                self.map.update({coordinate : temp_list})
                                

        #print("Result:", coordinate, "->", self.map[coordinate])
