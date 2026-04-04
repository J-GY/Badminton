from multiprocessing import Pool
import numpy as np
import networkx as nx
import pandas as pd
import yaml
import collections
import os
import random
import time
import sys
import numba 

random.seed(1998)
with open("config.yaml") as f:
    raw_text = f.read()
dataset = yaml.safe_load(raw_text)["dataset"]
raw_text = raw_text.format(dataset=dataset)
config = yaml.safe_load(raw_text)
dataset_point = config["pointnum"][str(config["dataset"])]

global_dist_matrix = None

def find_longest_trajectory():
    longest_traj = 0
    node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
    for node_list in node_list_int:
        if len(node_list) > longest_traj:
            longest_traj = len(node_list)
    return longest_traj

longest_traj_len = find_longest_trajectory()

def network_data():
    print("Loading road network...")
    rdnetwork = pd.read_csv('./data/{}/road/edge_weight.csv'.format(dataset), usecols=['section_id', 's_node', 'e_node', 'length'])
    
    roadnetwork_graph = nx.DiGraph()
    nx_vertice = pd.read_csv('./data/{}/road/node.csv'.format(dataset), usecols=['node'])
    roadnetwork_graph.add_nodes_from(nx_vertice['node'].tolist())

    for row in rdnetwork.values:
        roadnetwork_graph.add_edge(int(row[1]), int(row[2]), distance=row[-1])

    return None, None, None, None, None, None, roadnetwork_graph

def batch_Point_distance(roadnetwork):
    if os.path.exists('./ground_truth/{}/Full_Point_dis_matrix.npy'.format(dataset)):
        print("Full_Point_dis_matrix.npy exists, skipping point distance computation.")
        return

    print("Start computing Point Distance Matrix in batches...")
    pool = Pool(processes=20)
    for i in range(dataset_point + 1):
        if i != 0 and i % 1000 == 0:
            pool.apply_async(parallel_point_com, (i, list(range(i - 1000, i)), roadnetwork))
    pool.close()
    pool.join()

def parallel_point_com(i, id_list, roadnetwork):
    batch_list = []
    nodes_set = set(roadnetwork.nodes())
    
    for k in id_list:
        if k in nodes_set:
            length_list = nx.shortest_path_length(roadnetwork, source=k, weight='distance')
            one_list = np.full(dataset_point, -1.0, dtype=np.float32)

            for target, dist in length_list.items():
                if target < dataset_point:
                    one_list[target] = dist
            batch_list.append(one_list)
        else:
            batch_list.append(np.full(dataset_point, -1.0, dtype=np.float32))

    batch_list = np.array(batch_list, dtype=np.float32)
    p = './ground_truth/{}/'.format(dataset)
    if not os.path.exists(p):
        os.makedirs(p)
    np.save('./ground_truth/{}/Point_dis_matrix_{}.npy'.format(dataset, str(i)), batch_list)

def merge_Point_distance():
    save_path = './ground_truth/{}/Full_Point_dis_matrix.npy'.format(dataset)
    if os.path.exists(save_path):
        print("Loading all Point Distance Matrix from existing file...")
        return np.load(save_path)

    print("Merging Point Distance Matrices into Full_Point_dis_matrix.npy...")
    res = []
    max_range = ((dataset_point // 1000) + 1) * 1000
    
    for i in range(1000, max_range + 1001, 1000):
        file_path = './ground_truth/{}/Point_dis_matrix_{}.npy'.format(dataset, str(i))
        if os.path.exists(file_path):
            res.append(np.load(file_path))
    
    if len(res) > 0:
        full_matrix = np.concatenate(res, axis=0)
        full_matrix = full_matrix[:dataset_point, :dataset_point]
        
        np.save(save_path, full_matrix)
        print(f"Save to: {save_path}, Shape: {full_matrix.shape}")
        return full_matrix

        return None

def init_worker(matrix):
    global global_dist_matrix
    global_dist_matrix = matrix

@numba.jit(nopython=True)
def TP_dis_numba_strict(tr1, tr2, dist_matrix):
    M = tr1.shape[0]
    N = tr2.shape[0]
    
    # Loop 1: A -> B
    max1 = -1.0
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            node_i = tr1[i]
            node_j = tr2[j]

            dist = dist_matrix[node_i, node_j]

            if dist != -1:
                if dist < mindis:
                    mindis = dist
            else:
                return -1 
            
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    # Loop 2: B -> A
    max2 = -1.0
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            node_i = tr2[i]
            node_j = tr1[j]
            
            dist = dist_matrix[node_i, node_j]

            if dist != -1:
                if dist < mindis:
                    mindis = dist
            else:
                return -1
            
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1, max2))

def Traj_distance_worker(k, sample_list, test_list, valiortest):
    all_dis_list = []
    
    for sample in sample_list:
        one_dis_list = []
        sample_arr = np.array(sample, dtype=np.int32)
        
        for traj in test_list:
            traj_arr = np.array(traj, dtype=np.int32)
            dist = TP_dis_numba_strict(sample_arr, traj_arr, global_dist_matrix)
            one_dis_list.append(dist)
            
        all_dis_list.append(np.array(one_dis_list))

    all_dis_list = np.array(all_dis_list)
    
    p = './ground_truth/{}/{}_batch/'.format(dataset, valiortest)
    if not os.path.exists(p):
        os.makedirs(p)
    np.save(p + 'TP_spatial_distance_{}.npy'.format(str(k)), all_dis_list)

    return k

def batch_similarity_ground_truth(valiortest = None):
    print(f"Start computing {valiortest} Spatial Similarity...")
    node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)

    if valiortest == 'vali':
        target_list = node_list_int[int(config["train_set_size"]):int(config["train_set_size"])+int(config["vali_set_size"])]
    elif valiortest == 'test':
        target_list = node_list_int[int(config["train_set_size"])+int(config["vali_set_size"]):]
    else:
        print("Error: valiortest must be 'vali' or 'test'")
        return 0

    sample_list = target_list[:int(config["query_size"])]

    full_matrix = merge_Point_distance()
    
    results = []
    pool = Pool(processes=20, initializer=init_worker, initargs=(full_matrix,))
    
    batch_size = 50
    for i in range(batch_size, len(sample_list)+1, batch_size):
        batch_samples = sample_list[i-batch_size : i]
        
        r = pool.apply_async(Traj_distance_worker, (i, batch_samples, target_list, valiortest))
        results.append(r)
        
    pool.close()

    total = len(results)
    for idx, r in enumerate(results):
        try:
            k = r.get()
            if idx % 10 == 0:
                print(f"{idx}/{total} batch ({k}) completed.")
        except Exception as e:
            print(f"Batch Error: {e}")
            
    pool.join()
    return len(sample_list)

def merge_similarity_ground_truth(sample_len, valiortest):
    print(f"Merging {valiortest}...")
    res = []
    for i in range(50, sample_len+1, 50):
        file_path = './ground_truth/{}/{}_batch/TP_spatial_distance_{}.npy'.format(dataset, valiortest, str(i))
        if os.path.exists(file_path):
            res.append(np.load(file_path))
        else:
            print(f"Warning: missing {file_path}")
            
    if res:
        res = np.concatenate(res, axis=0)
        output_path = './ground_truth/{}/{}_spatial_distance.npy'.format(dataset, valiortest)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, res)
        print(f"Complete merging: {output_path}")

def generate_node_edge_interation():
    node_edge_dict = collections.defaultdict(set)
    edge = pd.read_csv('./data/{}/road/edge_weight.csv'.format(dataset))
    node_s, node_e = edge.s_node, edge.e_node

    for idx, (n_s, n_e) in enumerate(zip(node_s, node_e)):
        node_edge_dict[int(n_s)].add(idx)
        node_edge_dict[int(n_e)].add(idx)

    return node_edge_dict

def TP_dis(list_a, list_b):
    global global_dist_matrix

    if global_dist_matrix is None:
        global_dist_matrix = np.load('./ground_truth/{}/Full_Point_dis_matrix.npy'.format(dataset))

    arr_a = np.array(list_a, dtype=np.int32)
    arr_b = np.array(list_b, dtype=np.int32)

    return TP_dis_numba_strict(arr_a, arr_b, global_dist_matrix)

if __name__ == '__main__':
    _, _, _, _, _, _, road_net = network_data()
    
    batch_Point_distance(road_net)
    merge_Point_distance() 

    node_edge_dict = generate_node_edge_interation()

    sample_len = batch_similarity_ground_truth(valiortest='vali')
    merge_similarity_ground_truth(sample_len=sample_len, valiortest='vali')

    sample_len = batch_similarity_ground_truth(valiortest='test')
    merge_similarity_ground_truth(sample_len=sample_len, valiortest='test')
