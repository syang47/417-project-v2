import heapq
from copy import deepcopy
import math

def move(loc, dir):
    # added wait direction (0,0)
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values

def compute_he_heuristic(my_map, paths, i, old_heur):
    # iterate paths[i] against all other paths
    #   if it overlapps on a valid cell, change the value of the cell to large number

    # print(paths)
    m = len(my_map)
    n = len(my_map[1])
    age_num = i
    he_map = [ [1]*n for _ in range(m) ] 
    for i in range(m):
        for j in range(n):
            if my_map[i][j] == True:
                he_map[i][j] = math.inf

    if paths is not None:
        for curr_agent in range(len(paths)):
            for other_agent in range(len(paths)):
                if other_agent == curr_agent:
                    continue
                len1 = len(paths[curr_agent])
                len2 = len(paths[other_agent])
                for i in range(min(len1, len2)):
                    if paths[curr_agent][i] == paths[other_agent][i]:
                        he_map[paths[curr_agent][i][0]][paths[curr_agent][i][1]] += 1
    he_dict = dict()
    for i in range(m):
        for j in range(n):
            if he_map[i][j] != math.inf:
                print(old_heur, i)
                he_dict[(i,j)] = he_map[i][j] * 10 + old_heur[age_num][(i,j)]
    return he_dict

def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    constrant_table = {}
    for constraint in constraints:
        temp = deepcopy(constraint)
        time_step = temp['timestep']
        if constraint['timestep'] not in constrant_table:
            constrant_table[time_step] = []
        if constraint['agent'] != agent:   
            if 'positive' in temp and temp['positive']:
                if len(temp['loc']) != 1:
                    temp['loc'].reverse()
                    constrant_table[time_step].append({'loc':[temp['loc'][0],temp['loc'][0]]})
                temp['positive'] = False
                constrant_table[time_step].append(temp)  
        else:
            constrant_table[time_step].append(constraint)
    return constrant_table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    if next_time in constraint_table:
        for constraint in constraint_table[next_time]:
            if len(constraint['loc']) == 1 and constraint['loc'] == [next_loc]:
                    return True
            elif constraint['loc'] == [curr_loc, next_loc]:
                return True
            
        return False

    return False
    
    
def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node['he_val'], node))


def pop_node(open_list):
    _, _, _, _, curr = heapq.heappop(open_list)
    return curr

def push_node_to_focal(focal_list, node):
        heapq.heappush(focal_list, (node['he_val'] + node['g_val'], node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))
        # print("Generate node {}".format(self.num_of_generated))

def pop_node_from_focal(focal_list):        
    _, _, _, id, node = heapq.heappop(focal_list)
    # print("Expand node {}".format(id))
    return node

def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val'] 

def compare_focal_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['he_val'] < n2['he_val']

def a_star(my_map, start_loc, goal_loc, h_values, he_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.
    
    c_table = build_constraint_table(constraints, agent)
    open_list = []
    focal_list = []
    closed_list = dict()
    h_value = h_values[start_loc]
    he_value = he_values[start_loc]
    weight = 1

    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'he_val':he_value, 'timestep':0}
    push_node(open_list, root)
    push_node(focal_list, root)
    closed_list[(root['loc'], root['timestep'])] = root
    
    mapsize = len(my_map) * max([len(i) - sum(i) for i in my_map])
    
    cost_min = float('inf')
    while len(focal_list) > 0:
        curr = pop_node_from_focal(focal_list)

        for temp in open_list:

            if temp == curr:
                open_list.remove(temp)
        
        # check min and update cost min
        if len(open_list) > 0:
            new_min_node = pop_node(open_list)
            new_min = new_min_node['he_val']
            push_node(open_list, new_min_node)
            if new_min > cost_min:
                cost_min = new_min
                for i in open_list:

                    if i[-1]['cost'] < cost_min * weight and i not in focal_list:
                        print("testing")
                        push_node_to_focal(focal_list, i)

            
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc:
            # exceed longest path then stop
            
            max_timestep = 0
            flag = 0
            if curr['timestep'] >= mapsize:
                return None
            if c_table.keys():
                max_timestep = max(c_table.keys())
            if curr['timestep'] < max_timestep:

                for i in range(mapsize, curr['timestep'], -1):
                    if(is_constrained(goal_loc, goal_loc, i, c_table)):
                        flag = 1
                if flag == 0:
                    return get_path(curr) 
            else:
                return get_path(curr)         
            
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] < 0 or child_loc[1] < 0 or child_loc[0] >= len(my_map) or child_loc[1] >= len(my_map[0]): 
                continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue


            if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, c_table):
                continue

            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'he_val': he_values[child_loc],
                    'parent': curr,
                    'timestep':curr['timestep'] + 1}

                    
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
                    if( child['g_val'] + child['h_val'] < weight * cost_min):
                        push_node_to_focal(focal_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)
                if( child['g_val'] + child['h_val'] < weight * cost_min):
                        push_node_to_focal(focal_list, child)
    return None  # Failed to find solutions

