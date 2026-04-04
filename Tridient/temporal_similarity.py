from multiprocessing import Pool
import numpy as np
import time
import yaml
import os
import random
import numba 

with open("config.yaml") as f:
    raw_text = f.read()
dataset = yaml.safe_load(raw_text)["dataset"]
raw_text = raw_text.format(dataset=dataset)
config = yaml.safe_load(raw_text)

@numba.jit(nopython=True)
def TP_dis_numba_core(tr1, tr2):
    M = tr1.shape[0]
    N = tr2.shape[0]

    if M == 0 or N == 0:
        return -1

    # Loop 1: A -> B
    max1 = -1.0
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            d = np.abs(float(tr1[i]) - float(tr2[j]))
            if d < mindis:
                mindis = d
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    # Loop 2: B -> A
    max2 = -1.0
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            d = np.abs(float(tr2[i]) - float(tr1[j]))
            if d < mindis:
                mindis = d
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1, max2))

def TP_dis_wrapper(list_a, list_b):
    arr_a = np.array(list_a, dtype=np.int64)
    arr_b = np.array(list_b, dtype=np.int64)
    return TP_dis_numba_core(arr_a, arr_b)

def timelist_distance_worker(k, sample_list, test_list, valiortest):
    all_dis_list = []
    
    for sample in sample_list:
        one_dis_list = []
        for traj in test_list:
            dist = TP_dis_wrapper(sample, traj)
            one_dis_list.append(dist)
        
        all_dis_list.append(np.array(one_dis_list, dtype=np.float32))
        
    all_dis_list = np.array(all_dis_list)

    save_dir = './ground_truth/{}/{}_batch/'.format(dataset, valiortest)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    np.save(os.path.join(save_dir, 'TP_temporal_distance_{}.npy'.format(str(k))), all_dis_list)
    return k

def TP_dis(list_a, list_b):
    arr_a = np.array(list_a, dtype=np.int64)
    arr_b = np.array(list_b, dtype=np.int64)

    return TP_dis_numba_core(arr_a, arr_b)

def batch_timelist_ground_truth(valiortest=None):
    print(f"Start computing {valiortest} set...")
    time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
    if valiortest == 'vali':
        target_list = time_list_int[int(config["train_set_size"]):int(config["train_set_size"])+int(config["vali_set_size"])]
    elif valiortest == 'test':
        target_list = time_list_int[int(config["train_set_size"])+int(config["vali_set_size"]):]
    else:
        print("Error: valiortest must be 'vali' or 'test'")
        return 0

    sample_list = target_list[:int(config["query_size"])]
    
    pool = Pool(processes=20)
    results = []
    batch_size = 50

    for i in range(batch_size, len(sample_list) + 1, batch_size):
        batch_samples = sample_list[i-batch_size : i]

        r = pool.apply_async(
            timelist_distance_worker, 
            (i, batch_samples, target_list, valiortest)
        )
        results.append(r)
        
    pool.close()

    total = len(results)
    for idx, r in enumerate(results):
        try:
            k = r.get()
            if idx % 10 == 0:
                print(f"{idx}/{total} batch (batch_id={k}) completed.")
        except Exception as e:
            print(f"Batch Error: {e}")
            
    pool.join()
    print(f"{valiortest} computation completed.")
    return len(sample_list)

def merge_timelist_ground_truth(sample_len, valiortest):
    print(f"Merging {valiortest}...")
    res = []
    for i in range(50, sample_len + 1, 50):
        file_path = './ground_truth/{}/{}_batch/TP_temporal_distance_{}.npy'.format(dataset, valiortest, str(i))
        if os.path.exists(file_path):
            res.append(np.load(file_path))
        else:
            print(f"Warning: missing {file_path}")
    
    if res:
        final_res = np.concatenate(res, axis=0)
        save_path = './ground_truth/{}/{}_temporal_distance.npy'.format(dataset, valiortest)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, final_res)
        print(f"save to {save_path}, Shape: {final_res.shape}")

if __name__ == '__main__':
    sample_len = batch_timelist_ground_truth(valiortest='vali')
    merge_timelist_ground_truth(sample_len=sample_len, valiortest='vali')

    sample_len = batch_timelist_ground_truth(valiortest='test')
    merge_timelist_ground_truth(sample_len=sample_len, valiortest='test')