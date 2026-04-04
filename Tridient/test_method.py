import torch
import torch.nn.functional as F
import numpy as np
import yaml

def compute_embedding(net, road_network, test_traj, test_time, test_batch):

    if len(test_traj) <= test_batch:
        embedding = net(road_network, test_traj, test_time)
        return embedding
    else:
        i = 0
        all_embedding = []
        while i < len(test_traj):
            embedding = net(road_network, test_traj[i:i+test_batch], test_time[i:i+test_batch])
            all_embedding.append(embedding)
            i += test_batch

        all_embedding = torch.cat(all_embedding,0)
        return all_embedding

def test_model(embedding_set, isvali=False):
    with open("config.yaml") as f:
        raw_text = f.read()
    dataset = yaml.safe_load(raw_text)["dataset"]
    raw_text = raw_text.format(dataset=dataset)
    config = yaml.safe_load(raw_text)

    if isvali:
        input_dis_matrix = np.load(str(config["path_vali_truth"]))
    else:
        input_dis_matrix = np.load(str(config["path_test_truth"]))

    if isinstance(embedding_set, torch.Tensor):
        embedding_set = embedding_set.data.cpu().numpy()
    
    print(f"Embedding shape: {embedding_set.shape}")     
    print(f"GT Matrix shape: {input_dis_matrix.shape}") 

    l_recall_10 = 0
    l_recall_30 = 0
    l_recall_50 = 0
    l_recall_10_50 = 0
    f_num = 0

    for i in range(len(input_dis_matrix)):
        input_r = np.array(input_dis_matrix[i])
        valid_idx = [idx for idx, val in enumerate(input_r) if val != -1]
        if len(valid_idx) < 51:
            continue
        query_vec = embedding_set[i]
        dists = np.linalg.norm(embedding_set - query_vec, axis=1)
        
        input_r_valid = input_r[valid_idx]
        sorted_local_idx = np.argsort(input_r_valid)
        top50_gt_indices = [valid_idx[x] for x in sorted_local_idx[1:51]]
        top10_gt_indices = top50_gt_indices[:10]
        top30_gt_indices = top50_gt_indices[:30]

        pred_dist_valid = dists[valid_idx]
        sorted_pred_local_idx = np.argsort(pred_dist_valid)

        top50_pred_indices = [valid_idx[x] for x in sorted_pred_local_idx[1:51]]
        top10_pred_indices = top50_pred_indices[:10]
        top30_pred_indices = top50_pred_indices[:30]

        f_num += 1
        l_recall_10 += len(set(top10_gt_indices).intersection(set(top10_pred_indices)))
        l_recall_30 += len(set(top30_gt_indices).intersection(set(top30_pred_indices)))
        l_recall_50 += len(set(top50_gt_indices).intersection(set(top50_pred_indices)))
        l_recall_10_50 += len(set(top50_gt_indices).intersection(set(top10_pred_indices)))

        if i % 1000 == 0:
            print(f"Evaluated {i}/{len(input_dis_matrix)} queries...", end='\r')

    if f_num == 0: return 0,0,0,0

    recall_10 = float(l_recall_10) / (10 * f_num)
    recall_30 = float(l_recall_30) / (30 * f_num)
    recall_50 = float(l_recall_50) / (50 * f_num)
    recall_10_50 = float(l_recall_10_50) / (10 * f_num)

    return recall_10, recall_30, recall_50, recall_10_50
