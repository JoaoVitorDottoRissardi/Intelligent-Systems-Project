# Imports 
import numpy as np
import random

class GeneticTSP:

    def __init__(self, n_of_elem, initial_population, mutation_rate, dict, total_time, gravity_dict):
        self.n_of_elem = n_of_elem
        self.inital_population = initial_population
        self.mutation_rate = mutation_rate
        self.dict = dict
        self.total_time = total_time
        self.gravity_dict = gravity_dict

    #Offspring production
    def mutate_offspring(self,offspring):
        for q in range(int(self.n_of_elem*self.mutation_rate)):
            a = np.random.randint(0,self.n_of_elem)
            b = np.random.randint(0,self.n_of_elem)
            offspring[[a,b]] = offspring[[b,a]]

        return offspring
        
    # New populaiton generation
    def mutate_population(self,new_population_set):
        mutated_pop = []
        for offspring in new_population_set:
            mutated_pop.append(self.mutate_offspring(offspring))
        return mutated_pop

    # Pairs crossover
    def mate_progenitors(self,prog_a, prog_b):

        offspring1 = []
        offspring2 = []
        offspringTotal = []

        genA = int(random.random() * len(prog_a))
        genB = int(random.random() * len(prog_a))

        startGene = min(genA,genB)
        endGene = max(genA,genB)

        for i in range(startGene,endGene):
            offspring1.append(prog_a[i])

        offspring2 = [item for item in prog_b if item not in offspring1]

        offspringTotal = offspring1 + offspring2

        return np.array(offspringTotal)
                
        
    # Finding pairs of mates
    def mate_population(self,progenitor_list):
        new_population_set = []
        for i in range(progenitor_list.shape[1]):
            prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
            offspring = self.mate_progenitors(prog_a, prog_b)
            new_population_set.append(offspring)
            
        return new_population_set

    # Selecting the progenitors
    def progenitor_selection(self,population_set,fitness_list):
        
        total_fit = fitness_list.sum()
        prob_list = fitness_list/total_fit
        
        #Notice there is the chance that a progenitor. mates with oneself
        progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
        progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
        
        progenitor_list_a = population_set[progenitor_list_a]
        progenitor_list_b = population_set[progenitor_list_b]
        
        
        return np.array([progenitor_list_a,progenitor_list_b])

    def calculateVSG(self,can_save):
        vsg = 0
        for victim in can_save:
            if self.gravity_dict[victim] == '4':
                vsg += 1
            elif self.gravity_dict[victim] == '3':
                vsg += 2
            elif self.gravity_dict[victim] == '2':
                vsg += 3
            else:
                vsg += 6
        return vsg
    
    def calculate_last_part_fitness(self,elem_list,cost,current_index):
        non_taking_path_cost = 0
        while current_index < len(elem_list) - 1:
            non_taking_path_cost += cost[elem_list[current_index]][elem_list[current_index+1]]
            current_index += 1
        return non_taking_path_cost


    #individual solution
    
    def fitness_eval(self, elem_list, cost):
        can_save_cost = 0
        can_save = []

        current_index = 0

        can_save_cost += cost['0'][elem_list[current_index]]

        while(can_save_cost + cost[elem_list[current_index]]['0'] < self.total_time):
            can_save.append(elem_list[current_index])
            current_index += 1
            if current_index == len(elem_list):
                break
            can_save_cost += cost[elem_list[current_index-1]][elem_list[current_index]]


        
        if current_index == 0:
            return 1, []
        
        else:
            last_part_cost = self.calculate_last_part_fitness(elem_list,cost,current_index)
            vsg = self.calculateVSG(can_save)

            last_part_cost += 0.1

            fitness = (vsg^2)/(last_part_cost + can_save_cost)

            return fitness, can_save, vsg
        

    #All solutions
    def get_all_fitness(self,population_set, dict):
        fitness_list = np.zeros(self.inital_population)
        vsg_list = np.zeros(self.inital_population)
        can_save = []

        #Looping over all solutions computing the fitness for each solution
        for i in range(self.inital_population):
            fitness, victims, vsg = self.fitness_eval(population_set[i], dict)
            fitness_list[i] = fitness
            vsg_list[i] = vsg
            can_save.append(victims)

        return fitness_list, can_save, vsg_list

    # First step: Create the first population set
    def genesis(self, elem_list, n_population):

        population_set = []
        for i in range(n_population):
            #Randomly generating a new solution
            sol_i = elem_list[np.random.choice(list(range(self.n_of_elem)), self.n_of_elem, replace=False)]
            population_set.append(sol_i)
        return np.array(population_set)

    def calculate_solution(self):

        elem_list = []

        for elem in self.dict:
            elem_list.append(elem)

        elem_list.remove('0')

        elem_list = np.array(elem_list)
        population_set = self.genesis(elem_list, self.inital_population)
        fitness_list, can_save_list, vsg_list = self.get_all_fitness(population_set,self.dict)
        progenitor_list = self.progenitor_selection(population_set,fitness_list)
        new_population_set = self.mate_population(progenitor_list)
        mutated_pop = self.mutate_population(new_population_set)

        # Everything put together
        best_solution = [-1,0,np.array([])]
        for i in range(1000):
            fitness_list, can_save_list, vsg_list = self.get_all_fitness(mutated_pop,self.dict)

            #Saving the best solution
            if vsg_list.max() > best_solution[1]:

                print("Fitness: ", vsg_list.max())
                print("Generation : i", i)
                
                best_solution[0] = i
                best_solution[1] = vsg_list.max()
                best_index = list(vsg_list).index(vsg_list.max())
                best_solution[2] = can_save_list[best_index]
        
            progenitor_list = self.progenitor_selection(population_set,fitness_list)
            new_population_set = self.mate_population(progenitor_list)
        
            mutated_pop = self.mutate_population(new_population_set)

        return best_solution
