import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.animation as animation
from matplotlib.path import Path

from utils import *
from grid import *

def on_edge(point):
    x, y = point.x, point.y
    if x == 0 or x == 50 or y == 0 or y == 50:
        return True
    else:
        return False

#method that checks if a given point is in any of the turf polygons  
def isTurf(point, turfVertices):
    counter = 0
    for polygon in turfVertices:
        turfPath = Path(polygon)
        test = (point.x, point.y)
        
        if turfPath.contains_point(test, radius=-0.1) or on_edge(point):
            counter += 1
            
    if counter > 0:
        return True
    else:
        return False

#method that checks whether the given point is in any of the enclosures
def isEnclosed(point, enVertices):
    counter = 0
    for polygon in enVertices:
        enPath = Path(polygon)
        test = (point.x, point.y)
        
        if enPath.contains_point(test, radius=-0.1) or on_edge(point):
            counter += 1
            
    if counter > 0:
        return True
    else:
        return False
    
def print_to_summary(algoName, total, expanded):
    with open("summary.txt", "a") as f:
        print(algoName, file=f)
        print("Path cost:"+str(total), file=f)
        print("Nodes expanded:"+str(expanded), file=f)
        print()
        

#expands the nodes for BFS and DFS
def expand(point, goal, enVertices, reached):
    x, y = point.x, point.y   # extract x and y coordinates from node
    
    #up, right, down, left directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    children = []
    for dx, dy in directions:
        #add the displacement to the current coordinates to get the child node
        child_x, child_y = x + dx, y + dy
        child_node = Point(child_x, child_y)

        if isEnclosed(child_node, enVertices) or child_node in reached:
            continue
        else:
            heuristic = ((child_node.x - goal.x)**2 + (child_node.y - goal.y)**2)**0.5
            child_node.heuristic = heuristic
            child_node.parent = point
            children.append(child_node)
        
    return children

#expands the nodes for GBFS using SLD heuristic
def expand_best(point, goal, enVertices, reached):
    x, y = point.x, point.y   # extract x and y coordinates from node
    
    #up, right, down, left directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    children = []
    for dx, dy in directions:
        #add the displacement to the current coordinates to get the child node
        child_x, child_y = x + dx, y + dy
        child_node = Point(child_x, child_y)

        if isEnclosed(child_node, enVertices) or child_node in reached:
            continue
        else:
            heuristic = ((child_node.x - goal.x)**2 + (child_node.y - goal.y)**2)**0.5
            child_node.heuristic = heuristic
            child_node.parent = point
            children.append(child_node)      
        
        
    return children

#Expands the nodes for A* using SLD Heuristics
def expand_a(point, goal, turfVertices, enVertices, reached):
    x, y = point.x, point.y   # extract x and y coordinates from node
    
    #up, right, down, left directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    children = []
    for dx, dy in directions:
        #add the displacement to the current coordinates to get the child node
        child_x, child_y = x + dx, y + dy
        child_node = Point(child_x, child_y)

        #if it is in an enclosure, skip this child
        if isEnclosed(child_node, enVertices) or child_node in reached:
            continue
        
        #if its in a turf, calculate the corresponding heuristic
        if isTurf(child_node, turfVertices):
            #h(n)
            heuristic = ((child_node.x - goal.x)**2 + (child_node.y - goal.y)**2)**0.5
            #g(n)
            gn = (((point.x - child_node.x)**2 + (point.y - child_node.y)**2)**0.5) + point.heuristic
            child_node.heuristic = heuristic - gn
            child_node.parent = point
            children.append(child_node)
        else:
            heuristic = ((child_node.x - goal.x)**2 + (child_node.y - goal.y)**2)**0.5
            gn = (((point.x - child_node.x)**2 + (point.y - child_node.y)**2)**0.5) + point.heuristic
            child_node.heuristic = gn + heuristic
            child_node.parent = point
            children.append(child_node)           
        
        
    return children

#reconstructs that path from the path returned from the search algorithm
def reconstructPath(source, prev):
    path = []
    # Loop backwards through the array
    curr = prev[-1]
    path.append(curr)
    while curr != prev[0]:
        if curr is not None:
            curr = curr.parent
            path.append(curr)

    path.reverse()
    
    if path[0].__eq__(source):
        return path
    return []

#breadth first search algorithm which takes the source, dest, and the polygons to check
def breadth_first_search(source, dest, enVertices):
    algoName = 'bfs1'
    #keeps track of visited points
    reached = []
    
    total_cost = 0
    nodes_expanded = 0
    
    path = Stack()
    
    #source and dest points
    initNode = source
    goalNode = dest
    
    node = initNode #sets the first checked node to the start node
    node.heuristic = ((node.x - goalNode.x)**2 + (node.y - goalNode.y)**2)**0.5
    
    #if the init node is the goal, return the path
    if node.__eq__(goalNode):
        reached.append(node)
        total_cost = node.heuristic
        path.push(node)
        print_to_summary(algoName, total_cost, nodes_expanded)
        return path.list
    frontier = Queue() 
    frontier.push(node)
    
    reached.append(node)

    #while the frontier isn't empty, check the current node in frontier and
    #check each non-visited child expanded from the original node
    while not frontier.isEmpty():
        node = frontier.pop()
        nodes_expanded += 1
        total_cost += node.heuristic
        node.set_children(expand(node, goalNode, enVertices, reached))
        
        #for loop through children
        for child in node.children:
            if child.__eq__(goalNode):
                path.push(child)
                reached.append(child)
                finished = reconstructPath(source, path.list)
                print_to_summary(algoName, total_cost, nodes_expanded)
                return finished
            if child not in reached:
                reached.append(child)
                frontier.push(child)
                path.push(node)

    return []

#DFS program that takes the start, end point, and the enclosures
def depth_first_search(source, dest, enVertices):
    frontier = Stack()
    
    total_cost = 0
    nodes_expanded = 0
    
    path = Stack()
    reached = []
    
    initNode = source
    goalNode = dest
    node = initNode
    node.heuristic = ((node.x - goalNode.x)**2 + (node.y - goalNode.y)**2)**0.5
    
    frontier.push(node)
    reached.append(node)
    
    while not frontier.isEmpty():
        node = frontier.pop()
        total_cost += node.heuristic
        node.set_children(expand(node, goalNode, enVertices, reached))
        if node.__eq__(goalNode):
            print_to_summary("dfs1", total_cost, nodes_expanded)
            path.push(node)
            finished = reconstructPath(source, path.list)
            return finished
        for child in node.children:
            nodes_expanded += 1
            if child not in reached:
                frontier.push(child)
                reached.append(child)
                path.push(node)
    
    return []

#method that checks whether or not the cost is less than all the other costs in reached
def less_than_pathcost(reached, cost):
     
    #traverse in the list
    for node in reached:
        #compare with all the values with cost
        if cost>= node.heuristic:
            return False
    return True

#Greedy Best-First Search Algorithm
def greedy_bfs(source, dest, enVertices):
    initNode = source
    goalNode = dest
    node = initNode
    heuristic = ((node.x - goalNode.x)**2 + (node.y - goalNode.y)**2)**0.5
    node.heuristic = heuristic
    
    total_cost = 0
    nodes_expanded = 0
    
    path = Stack()
    
    frontier = PriorityQueue()
    frontier.update(node, node.heuristic)

    path.push(node)
    
    reached = Stack()
    reached.push(node)
    
    while not frontier.isEmpty():
        node = frontier.pop()
        total_cost += 1
        node.set_children(expand_best(node, goalNode, enVertices, reached.list))
        
        if node.__eq__(goalNode):
            path.push(node)
            print_to_summary("gbfs1", total_cost, nodes_expanded)
            return reconstructPath(source, path.list)
        nodes_expanded += 1
        for child in node.children:
            if child not in reached.list or less_than_pathcost(reached.list, child.heuristic):
                reached.push(child)
                frontier.update(child, child.heuristic)
                path.push(node)
    
    return []

#A* Algorithm
def a_star(source, dest, turfVertices, enVertices):
    initNode = source
    goalNode = dest
    node = initNode
    heuristic = ((node.x - goalNode.x)**2 + (node.y - goalNode.y)**2)**0.5
    node.heuristic = heuristic
    
    path = Stack()
    
    frontier = PriorityQueue()
    frontier.update(node, node.heuristic)

    path.push(node)
    
    reached = Stack()
    reached.push(node)
    
    total_cost = 0
    nodes_expanded = 0
    
    while not frontier.isEmpty():
        node = frontier.pop()
        total_cost+= 1
        node.set_children(expand_a(node, goalNode, turfVertices, enVertices, reached.list))
        
        if node.__eq__(goalNode):
            path.push(node)
            print_to_summary("astar", total_cost, nodes_expanded)
            return reconstructPath(source, path.list)
        nodes_expanded += 1
        for child in node.children:
            if child not in reached.list or less_than_pathcost(reached.list, child.heuristic):
                reached.push(child)
                frontier.update(child, child.heuristic)
                path.push(node)
    
    return []

def gen_polygons(worldfilepath):
    polygons = []
    with open(worldfilepath, "r") as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
        for line in lines:
            polygon = []
            pts = line.split(';')
            for pt in pts:
                xy = pt.split(',')
                polygon.append(Point(int(xy[0]), int(xy[1])))
            polygons.append(polygon)
    return polygons

if __name__ == "__main__":
    epolygons = gen_polygons('TestingGrid/world1_enclosures.txt')
    tpolygons = gen_polygons('TestingGrid/world1_turfs.txt')
    
    #epolygons = gen_polygons("TestingGrid/vincent_enclosures.txt")
    #epolygons = gen_polygons("TestingGrid/vincent_enclosures2.txt")
    #tpolygons = gen_polygons("TestingGrid/vincent_turfs.txt")

    enVertices = []
    
    turfVertices = []
    
    source = Point(8,10)
    dest = Point(43,45)
    
    #source = Point(10, 3)
    #dest = Point(34, 38)

    fig, ax = draw_board()
    draw_grids(ax)
    draw_source(ax, source.x, source.y)  # source point
    draw_dest(ax, dest.x, dest.y)  # destination point
    
    # Draw enclosure polygons
    for polygon in epolygons:
        eachEnPolyVertices = []
        for p in polygon:
            eachEnPolyVertices.append((p.x, p.y))
            #enVertices.append((p.x, p.y))
            draw_point(ax, p.x, p.y)
        enVertices.append(eachEnPolyVertices)
        
    for polygon in epolygons:
        for i in range(0, len(polygon)):
            draw_line(ax, [polygon[i].x, polygon[(i+1)%len(polygon)].x], [polygon[i].y, polygon[(i+1)%len(polygon)].y])
    
    # Draw turf polygons
    for polygon in tpolygons:
        eachTurfPolyVertices = []
        for p in polygon:
            eachTurfPolyVertices.append((p.x, p.y))
            draw_green_point(ax, p.x, p.y)
        turfVertices.append(eachTurfPolyVertices)    
        
    for polygon in tpolygons:
        for i in range(0, len(polygon)):
            draw_green_line(ax, [polygon[i].x, polygon[(i+1)%len(polygon)].x], [polygon[i].y, polygon[(i+1)%len(polygon)].y])

    #### Here call your search to compute and collect res_path 
        
    while True:
        try:
            user_input = int(input("1.BFS\n2.DFS\n3.GBFS\n4.A*\n\nYour Choice: "))
            if user_input == 1:
                res_path = breadth_first_search(source, dest, enVertices)
                break
            elif user_input == 2:
                res_path = depth_first_search(source, dest, enVertices)
                break
            elif user_input == 3:
                res_path = greedy_bfs(source, dest, enVertices)
                break
            elif user_input == 4:
                res_path = a_star(source, dest, turfVertices, enVertices)
                break
            else:
                print("Please try again")
        except ValueError:
            print("Enter Only Integer Values.")

    with open('summary.txt', 'r+',) as f:
        f.truncate(0)

    bfs = breadth_first_search(source, dest, enVertices)
    dfs = depth_first_search(source, dest, enVertices)
    gbfs = greedy_bfs(source, dest, enVertices)
    astar = a_star(source, dest, turfVertices, enVertices)
        
        
    
    #res_path = [Point(24,17), Point(25,17), Point(26,17), Point(27,17),  
                #Point(28,17), Point(28,18), Point(28,19), Point(28,20)]
    
    for i in range(len(res_path)-1):
        print(res_path[i])
        draw_result_line(ax, [res_path[i].x, res_path[i+1].x], [res_path[i].y, res_path[i+1].y])
        plt.pause(0.1)
    
    plt.show()
    plt.close()
