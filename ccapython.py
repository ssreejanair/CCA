import networkx as nx
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


data = pd.read_csv('')
data1 =data[['source','sourceL']]
data2 =data[['target','targetL']]
data1.columns = ['domain','label']
data2.columns = ['domain','label']
nodelist = data1.append(data2)
nodelist = nodelist.drop_duplicates()
nodelist['label'][nodelist['label'] == 0]='l'
nodelist['label'][nodelist['label'] == 1]='r'

node_distribution = nodelist['label'].value_counts()
node_distribution

edge = data[['source','target']]
graph = nx.from_pandas_edgelist(edge)
graph.remove_edges_from(nx.selfloop_edges(graph))
print(graph.number_of_nodes())
print(graph.number_of_edges())
edge =nx.to_pandas_edgelist(graph)
nodelist.columns =['node','label']

node_distribution = nodelist['label'].value_counts()
node_distribution


def get_node_label(nodelist, node):
    return nodelist[nodelist['node'] == node]['label'].values[0]

def node_data(nodelist, edgelist, LABELS):
    G = nx.from_pandas_edgelist(edgelist, 'source', 'target')
    LABELS.sort()
    columns = ['node', 'label', 'total']
    for lbl in LABELS:
        columns.append(lbl)

    all_data = []
    for node in G.nodes():
        cluster_node_count = dict.fromkeys(LABELS, 0)
        total = 0
        for neighbhor in G.neighbors(node):
            total += 1
            neigh_lbl =  get_node_label(nodelist, neighbhor)
            cluster_node_count[neigh_lbl] += 1
       
        label =  get_node_label(nodelist, node)
        data = [node, label, total]
        keys = list(cluster_node_count.keys())
        keys.sort()
        for key in keys:
            data.append(cluster_node_count[key])
       
        all_data.append(data)
   
    info = pd.DataFrame(all_data, columns=columns)
    return info


def neighbor_contribution(node, node_label, neighbor, info, LABELS):
    wc_i = W[node_label]
   
    nc = 0
    total = info.loc[info['node'] == neighbor , 'total'].iloc[0]
   
    for lbl in LABELS:
        count = info.loc[info['node'] == neighbor , lbl].iloc[0]
        count = count - 1 if lbl == node_label else count
        nc += ((1 / 2) * wc_i[lbl] * count / (total - 1)) if total > 1 else 0
       
    return nc


def avg_neighbor_contribution(graph, node, node_label, info, cluster_label, LABELS):
    nc_total = 0
    total = 0
   
    for neighbor in graph.neighbors(node):
        neigh_label = info.loc[info['node'] == neighbor , 'label'].iloc[0]
        if neigh_label != cluster_label:
            continue
       
        nc_total += neighbor_contribution(node,  node_label, neighbor, info, LABELS)
        total += 1
   
    return  nc_total / total if total > 0 else 0
   
def ideology_factor(graph, node, node_label, info, cluster_label, LABELS):
    wc_i = W[node_label]
    count = info.loc[info['node'] == node, cluster_label].iloc[0]
    total = info.loc[info['node'] == node, 'total'].iloc[0]
    id_factor = wc_i[cluster_label] * count / total    
    return id_factor



def cluster_polarization(cluster):
    total_nodes = len(cluster)
    sum_of_border_nodes = sum(cluster['H'])
    cluster_affinity = sum_of_border_nodes / total_nodes
    return cluster_affinity

def heterophily(nodelist, edgelist):
    LABELS = list(set(edgelist['source_label']).union(set(edgelist['target_label'])))
    node_info = node_data(nodelist, edgelist, LABELS)
    G = nx.from_pandas_edgelist(edgelist, 'source', 'target')

    data = pd.DataFrame([])
    for node in G.nodes():
        I = 0
        NC = 0
        non_zero_cluster_count = 0
        node_label = node_info.loc[node_info['node'] == node , 'label'].iloc[0]
        details = node_info.loc[node_info['node'] == node]
        for lbl in LABELS:
            nc = avg_neighbor_contribution(G, node, node_label, node_info, lbl, LABELS)
            if nc != 0:
                NC += nc
                non_zero_cluster_count += 1
            I += ideology_factor(G, node, node_label, node_info, lbl, LABELS)
      #
       
        H =  (NC / non_zero_cluster_count if non_zero_cluster_count > 0 else 0) + I
 
       
        details['H'] =H
        #details = details.drop(details.index[[0]])
        #print(details)
        data = data.append(details)
        #print(data)
   
    H_df = pd.DataFrame(data)
    return H_df


W = {
    'r':{'r':-1, 'l':1},
    'l':{'r':1,'l':-1}
}

originalnodelist = nodelist.copy(deep=True)
originaledgelist = edge.copy(deep=True)
