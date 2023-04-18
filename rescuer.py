##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim

from cmath import inf
from math import cos
import os
import random
import numpy as np
from urllib.parse import MAX_CACHE_SIZE
from abstract_agent import AbstractAgent
from physical_agent import PhysAgent
from abc import ABC, abstractmethod
from collections import deque
from graph import Graph
from genetic import GeneticTSP


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstractAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.plan = []              # a list of planned actions
        self.rtime = self.TLIM      # for controlling the remaining time
        self.map = {}
        self.victims = []

        self.saved_list = []
        self.victims_to_save = []

        self.solution_cost = 0

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.body.set_state(PhysAgent.IDLE)

        # planning
        #self.__planner()
    
    def go_save_victims(self, map, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""
        self.body.set_state(PhysAgent.ACTIVE)
        self.map = map
        self.victims = victims
        self.__planner()
        
    
    def __planner(self):
        """ A private method that calculates the walk actions to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        # This is a off-line trajectory plan, each element of the list is
        # a pair dx, dy that do the agent walk in the x-axis and/or y-axis

        if len(self.victims) == 0:
            return

        # Gravity dict calculation

        gravity_dict = {}

        for victim in self.victims:
            gravity_dict.update({victim[1][0] : victim[1][7]})

        #print(gravity_dict)

        # Calculate cost and path to travel from each victim to another

        graph = Graph(self.map)

        cost_dict = {}
        path_dict = {}

        for victim_1 in self.victims:
            dict1 = {}
            dict2 = {}
            for victim_2 in self.victims:
                if victim_1[1][0] != victim_2[1][0]:
                    path = graph.a_star_search(victim_1[0], victim_2[0])
                    path.pop()
                    path_cost = self.calculate_path_cost(path)
                    dict1.update({victim_2[1][0] : path_cost + self.COST_FIRST_AID})
                    dict2.update({victim_2[1][0] : path})
            path = graph.a_star_search(victim_1[0], (0,0))
            path.pop()
            path_cost = self.calculate_path_cost(path)
            dict1.update({'0' : path_cost})
            dict2.update({'0' : path})
            cost_dict.update({victim_1[1][0] : dict1})
            path_dict.update({victim_1[1][0] : dict2})

        dict1 = {}
        dict2 = {}
        
        for victim in self.victims:
            path = graph.a_star_search((0,0), victim[0])
            path.pop()
            path_cost = self.calculate_path_cost(path)
            dict1.update({victim[1][0] : path_cost + self.COST_FIRST_AID})
            dict2.update({victim[1][0] : path})

        cost_dict.update({'0' : dict1})
        path_dict.update({'0' : dict2})

        solution_path = []
        number_of_victims = len(self.victims)

        # for key in cost_dict:
        #     print(key, cost_dict[key])

        #print(cost_dict)

        #Tries to save the maximum number of victim with the time it haves

        print("Initiating genetic algorithm calculations")

        gen = GeneticTSP(number_of_victims, 500, 0.05, cost_dict, self.TLIM, gravity_dict)

        solution = gen.calculate_solution()

        print(solution)

        solution_list = solution[2]

        if solution_list == []:
            return

        #print(solution_list)

        #print("Solution order: ", solution_list)

        self.solution_cost = cost_dict['0'][solution_list[0]]

        for i in range(len(solution_list) -1):
            self.solution_cost += cost_dict[solution_list[i]][solution_list[i+1]]

        self.solution_cost += cost_dict[solution_list[len(solution_list) - 1]]['0']      

        solution_path = path_dict['0'][solution_list[0]]
        solution_path.pop()

        for i in range(len(solution_list) -1):
            current_path = path_dict[solution_list[i]][solution_list[i+1]]
            current_path.pop()
            solution_path = solution_path + current_path

        solution_path = solution_path + path_dict[solution_list[len(solution_list) - 1]]['0']

        self.victims_to_save = [int(x)-1 for x in solution_list]

        #print("Solution Path: ", solution_path)

        current_position = (0,0)
        solution_path.pop(0)    

        #print(solution_cost)

        while len(solution_path) > 0:
            (dx, dy) = (solution_path[0][0] -  current_position[0], solution_path[0][1] - current_position[1])
            current_position = solution_path.pop(0)
            self.plan.append((dx, dy))
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)

        # Walk - just one step per deliberation
        result = self.body.walk(dx, dy)

        # Rescue the victim at the current position
        if result == PhysAgent.EXECUTED:
            # check if there is a victim at the current position
            seq = self.body.check_for_victim()
            if seq >= 0 and seq in self.victims_to_save and seq not in self.saved_list:
                res = self.body.first_aid(seq) # True when rescued 
                self.saved_list.append(seq)
            elif seq >= 0 and seq not in self.saved_list and self.solution_cost + self.COST_FIRST_AID <= self.TLIM:
                print("Saving not pretended: ", seq)        
                res = self.body.first_aid(seq)
                self.saved_list.append(seq)
                self.solution_cost += self.COST_FIRST_AID

        return True
    
    def calculate_path_cost(self, path):

        coordinate_1 = path[0]

        total_cost = 0

        for i in range(1, len(path)):

            coordinate_2 = path[i]
            (dx, dy) = (coordinate_2[0] - coordinate_1[0], coordinate_2[1] - coordinate_1[1])

            if dx == 0 or dy == 0:
                total_cost += self.COST_LINE
            else:
                total_cost += self.COST_DIAG

            coordinate_1 = coordinate_2

        return total_cost

