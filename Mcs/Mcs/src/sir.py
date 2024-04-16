import random
import networkx as nx
import re
import os
import pandas as pd
import time


def readEdgeslistToGraph(edges_filename):
    edge = []
    with open(edges_filename, 'r', encoding='utf-8-sig') as f:
        data = f.readlines()
        for line in data:
            line_str = line.replace('\r', '').replace('\n', '').replace('\t', '')
            line = re.split(',| ', line_str)
            single_edge = tuple([line[0], line[1]])
            edge.append(single_edge)
    G = nx.Graph()
    G.add_edges_from(edge)
    return G


def randomChoiseInfectedNodes(G, initial_infected_nodes_num):
    infected_nodes = random.sample(list(G.nodes), k=initial_infected_nodes_num)
    for node in G.nodes():
        G.nodes[node]["state"] = "S"
    for node in infected_nodes:
        G.nodes[node]["state"] = "I"


def updateNodeState(G, node, infected_rate, recover_rate, old_G):
    if old_G.nodes[node]["state"] == "I":
        p = random.random()
        if p < recover_rate:
            G.nodes[node]["state"] = "R"
    elif old_G.nodes[node]["state"] == "S":
        k = 0
        for neibor in old_G.adj[node]:
            if old_G.nodes[neibor]["state"] == "I":
                k += 1
        p = random.random()
        if p < (1 - (1 - infected_rate) ** k):
            G.nodes[node]["state"] = "I"


def updateNetworkState(G, infected_rate, recover_rate):
    old_G = G.copy()
    for node in G:
        updateNodeState(G, node, infected_rate, recover_rate, old_G)


def countSIRnum(G):
    S = 0
    I = 0
    for node in G:
        if G.nodes[node]["state"] == "S":
            S += 1
        if G.nodes[node]["state"] == "I":
            I += 1
    R = len(G.nodes) - S - I
    return S, I, R


def eachIterateSIRnum(days):
    eachiterate_SIR_list = []
    for day in range(1, days + 1):
        updateNetworkState(G, infected_rate, recover_rate)
        tuple_sir = countSIRnum(G)
        # print("day%s:\tS:%s\tI:%s\tR:%s"%(day,tuple_sir[0],tuple_sir[1],tuple_sir[2]))
        eachiterate_SIR_list.append(list(tuple_sir))
    return eachiterate_SIR_list


if __name__ == '__main__':

    start = time.time()

    edges_filename = 'l_1.txt'
    initial_infected_nodes_num = 10
    infected_rate = 0.1
    recover_rate = 0.02
    days = 100
    iterate_num = 100

    average_SIR_list = [[0, 0, 0] for i in range(days)]
    for i in range(iterate_num):
        G = readEdgeslistToGraph(edges_filename)
        randomChoiseInfectedNodes(G, initial_infected_nodes_num)
        eachiterate_SIR_list = eachIterateSIRnum(days)
        for day in range(days):
            for state in range(3):
                average_SIR_list[day][state] += eachiterate_SIR_list[day][state]
    for day in range(days):
        for state in range(3):
            average_SIR_list[day][state] = average_SIR_list[day][state] / iterate_num

    end = time.time()
