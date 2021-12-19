from copy import deepcopy
import math
import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost, compute_he_heuristic


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    max_path = max(len(path1), len(path2))

    for i in range(max_path):
        if get_location(path1, i) == get_location(path2, i):
            return [[get_location(path1, i)], i]

        if i  > 0 and i < max_path-1:
             if get_location(path1, i) == get_location(path2, i+1) and get_location(path2, i) == get_location(path1, i+1):
                return [[get_location(path1, i), get_location(path1, i+1)], i+1]

    # no collision between two paths
    return None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    # detected_collision = detect_collision(paths[0], paths[1])
    collisions = []
    for i in range(len(paths)-1):
        for j in range(i+1, len(paths)):
            detected_col = detect_collision(paths[i], paths[j])
            if detected_col != None:
                collisions.append({
                    'a1': i, 
                    'a2': j, 
                    'loc': detected_col[0],
                    'timestep': detected_col[1]
                })
    return collisions

def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timeste

    # vertex collision
    if len(collision['loc']) == 1:
        first = {
            'agent': collision['a1'],
            'loc': collision['loc'], 
            'timestep': collision['timestep']
        }
        second = {
            'agent': collision['a2'], 
            'loc': collision['loc'], 
            'timestep': collision['timestep']
        }
    # edge collision
    else:
        first_loc = collision['loc'][0]
        sec_loc = collision['loc'][1]
        first = {
            'agent': collision['a1'], 
            'loc': [first_loc,sec_loc], 
            'timestep': collision['timestep']
        }
        second = {
            'agent': collision['a2'], 
            'loc': [sec_loc,first_loc], 
            'timestep': collision['timestep']
        }
    return [first, second]

    
def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    agent = random.randint(0, 1)
    if len(collision['loc']) == 1:
        if agent == 1:
            first = {
                'agent': collision['a1'], 
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': True
            }
            second = {
                'agent': collision['a1'], 
                'loc': collision['loc'],
                'timestep': collision['timestep'], 
                'positive': False
            }
        else:
            first = {
                'agent': collision['a2'], 
                'loc': collision['loc'],
                'timestep': collision['timestep'],
                'positive': True
            }
            second = {
                'agent': collision['a2'], 
                'loc': collision['loc'],
                'timestep': collision['timestep'],'positive': False
            }
    # edge collision
    else:
        if agent == 1:
            first = {
                'agent': collision['a1'], 'loc': [collision['loc'][0], collision['loc'][1]],
                'timestep': collision['timestep'],
                'positive': True
            }
            second = {
                'agent': collision['a1'], 'loc': [collision['loc'][0], collision['loc'][1]],
                'timestep': collision['timestep'], 
                'positive': False
            }
        else:
            first = {
                'agent': collision['a2'], 'loc': [collision['loc'][1], collision['loc'][0]], 
                'timestep': collision['timestep'],
                'positive': True
            }
            second = {
                'agent': collision['a2'], 'loc': [collision['loc'][1], collision['loc'][0]], 
                'timestep': collision['timestep'], 
                'positive': False
            }
    return [first, second]

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst

class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.num_of_focal_generated = 0
        self.num_of_focal_expanded = 0
        self.CPU_time = 0

        self.next = 0
        self.weight = 1
        self.focal_list = []
        self.open_list = []
        self.cost_min = float('inf')

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

        self.heuristics_perpath = []
        for goal in self.goals:
            self.heuristics_perpath.append(compute_he_heuristic(None, 0))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1
    
    def push_node_to_focal(self, node):
        heapq.heappush(self.focal_list, (node['heuristic_constraint'], node['cost'], len(node['collisions']), self.num_of_focal_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_focal_generated += 1

    def pop_node(self):
        _, _, _, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def pop_node_from_focal(self):
        _, _, _, _, node = heapq.heappop(self.focal_list)
        # print("Expand node {}".format(id))
        self.num_of_focal_expanded += 1
        return node

    def h2(self, paths, num_agents):
        k = 0
        collision = detect_collisions(paths)
        # print("collision: ", collision)
        conflicted_agents = []
        for i in range(len(collision)):
            # print(collision[i]['a1'])
            if collision[i]['a1'] not in conflicted_agents:
                conflicted_agents.append(collision[i]['a1'])
        k = len(conflicted_agents)

        # for current_agent in range(num_agents-1):
        #     for other_agent in range(current_agent+1, num_agents-1):
        #         collision = detect_collision(paths[current_agent], paths[other_agent])
        #         if collision:
        #             k += 1
        #             continue
        # print(k)
        return k

    def h1(self, collisions):
        return len(collisions)


    def h3(self, next, collision, paths, num_agents):
        if next % 2 == 0:
            return self.h1(collision)
        else:
            return self.h2(paths, num_agents)
        # pass
    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': [],
                'heuristic_constraint': 0} # added heuristic constraint
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], self.heuristics_perpath[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        self.cost_min = root['cost']    # cost_min

        root['heuristic_constraints'] = len(root['collisions']) # h_c function 

        self.push_node_to_focal(root)

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        ## Apply Focal Search
        while self.focal_list:
            curr = self.pop_node_from_focal()

            # remove curr from open
            for i in self.open_list:
                if i[-1] == curr:
                    self.open_list.remove(i)

            # if no collision return solution
            if len(curr['collisions']) == 0:
                self.print_results(curr)
                return curr['paths']

            
            collision = curr['collisions'][0]
            if disjoint:
                constraints = disjoint_splitting(collision)
            else:
                constraints = standard_splitting(collision)

            for constraint in constraints:
                new_constraint = deepcopy(curr['constraints'])
                child_path = deepcopy(curr['paths'])
                child = {'cost': 0,
                        'constraints': new_constraint + [constraint],
                        'paths': child_path,
                        'collisions': [],
                        'heuristic_constraint': 0}
                if 'positive' in constraint and constraint['positive']:
                    agents = paths_violate_constraint(constraint, child_path)
                else:
                    agents = [constraint['agent']]

                # print("List of conflict agent", agents)
                for i in agents:
                    # print("Calling A* for", i)
                    self.heuristics_perpath[i] = compute_he_heuristic(child['paths'], i)
                    path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], self.heuristics_perpath[i],
                            i, child['constraints'])
                    child['paths'][i] = path
                    if not path:
                        break

                if path is not None:  
                    child['collisions'] = detect_collisions(child['paths'])
                    child['cost'] = get_sum_of_cost(child['paths'])
                    # child['heuristic_constraints'] = self.h1(child['collisions'] )
                    # child['heuristic_constraints'] = self.h2(path, self.num_of_agents)
                    child['heuristic_constraints'] = self.h3(self.next, child['collisions'], path, self.num_of_agents)
                    self.next = (self.next + 1) % 2

                    self.push_node(child)
                    if child['cost'] <= self.weight *  self.cost_min:
                        self.push_node_to_focal(child)
            
            if(len(self.open_list) > 0):
                new_min_node = self.pop_node()
                new_min = new_min_node['cost']
                self.push_node(new_min_node)
                if new_min > self.cost_min:
                    self.cost_min = new_min
                    for node in self.open_list:
                        if node[-1]['cost'] <= self.cost_min * self.weight:
                            self.push_node_to_focal(node[-1])
                        
        raise BaseException('No solutions')


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
