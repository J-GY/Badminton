import yaml
import torch
from model_network import DualStreamEncoder
import data_utils
from lossfun import LossFun
import test_method
import time
import os

class TRIDENT_Trainer(object):
    def __init__(self):
        with open("config.yaml") as f:
            raw_text = f.read()
        dataset = yaml.safe_load(raw_text)["dataset"]
        raw_text = raw_text.format(dataset=dataset)
        config = yaml.safe_load(raw_text)
        self.feature_size   = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.date2vec_size  = config["date2vec_size"]
        self.hidden_size    = config["hidden_size"]
        self.num_layers     = config["num_layers"]
        self.device         = torch.device(
            f"cuda:{config['cuda']}" if torch.cuda.is_available() else "cpu"
        )
        self.attn_temperature = config["attn_temperature"]

        self.learning_rate  = config["learning_rate"]
        self.epochs         = config["epochs"]
        self.self_layer     = config["self_layer"]
        self.train_batch    = config["train_batch"]
        self.test_batch     = config["test_batch"]
        self.dataset        = config["dataset"]
        self.early_stop     = config["early_stop"]

        self.best_hr10 = 0  
        self.best_epoch = 0  
        self.no_improvement_count = 0

    def _build_model(self):
        return DualStreamEncoder(
            feature_size=self.feature_size,
            date2vec_size=self.date2vec_size,
            hid_size=self.embedding_size,
            nhead=4,
            ffn_dim=self.embedding_size * 4,
            num_layers=self.num_layers,
            device=self.device,
            num_self_layers=self.self_layer,
            attn_temperature=self.attn_temperature,
            d_drop=0.1
        ).to(self.device)

    def TRIDENT_eval(self, load_model=None):
        print("Evaluation")
        print(load_model)
        net = DualStreamEncoder(
            feature_size=self.feature_size,
            date2vec_size=self.date2vec_size,
            hid_size=self.embedding_size,
            nhead=4,
            ffn_dim=self.embedding_size * 4,
            num_layers=self.num_layers,
            device=self.device,
            num_self_layers=self.self_layer,
            attn_temperature=self.attn_temperature,
            d_drop=0
        ).to(self.device)

        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)

            dataload = data_utils.DataLoader()
            road_network = data_utils.load_netowrk(self.dataset).to(self.device)

            with torch.no_grad():
                vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='test')
                embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                               test_traj=list(vali_node_list),
                                                               test_time=list(vali_d2vec_list),
                                                               test_batch=self.test_batch)
                acc = test_method.test_model(embedding_vali, isvali=False)
                print(acc)

    def TRIDENT_train(self, load_model=None, load_optimizer=None):
        net = self._build_model()
        if isinstance(load_model, str):
            net.load_state_dict(
                torch.load(load_model, map_location=self.device)
            )

        optimizer = torch.optim.Adam(
            [p for p in net.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        if isinstance(load_optimizer, str):
            optimizer.load_state_dict(
                torch.load(load_optimizer, map_location=self.device)
            )

        loss_fn = LossFun(self.train_batch).to(self.device)

        net.to(self.device)

        dataload = data_utils.DataLoader()
        dataload.get_triplets()
        data_utils.triplet_groud_truth()
        road_network = data_utils.load_netowrk(self.dataset).to(self.device)

        bt_num = dataload.return_triplets_num() // self.train_batch
        batch_l = data_utils.batch_list(batch_size=self.train_batch)
        total_train_time = 0
        total_predict_time = 0
        for epoch in range(self.epochs):
            net.train()
            t0 = time.time()
            for _ in range(bt_num):
                a_n, a_t, p_n, p_t, n_n, n_t, idx = batch_l.getbatch_one()
                a_e = net(road_network, a_n, a_t)
                p_e = net(road_network, p_n, p_t)
                n_e = net(road_network, n_n, n_t)
                loss = loss_fn(a_e, p_e, n_e, idx)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t1 = time.time()
            print(f"[Epoch {epoch}] train_time={t1-t0:.1f}s loss={loss.item():.4f}")
            total_train_time += (t1 - t0)


            if epoch % 2 == 0:
                net.eval()
                with torch.no_grad():
                    t0_valid = time.time()
                    v_n, _, v_t = dataload.load(load_part='vali')
                    emb_v = test_method.compute_embedding(
                        road_network=road_network,
                        net=net,
                        test_traj=list(v_n),
                        test_time=list(v_t),
                        test_batch=self.test_batch
                    )
                    hr10, hr30, hr50, hr1050 = test_method.test_model(
                        emb_v, isvali=True
                    )
                    t1_valid = time.time()
                    total_predict_time += (t1_valid - t0_valid)
                    print(f" Val HR10={hr10:.4f} HR30={hr30:.4f} HR50={hr50:.4f} HR1050={hr1050:.4f}")

                    fn = (
                        f"./model/{self.dataset}/"
                        f"epoch{epoch}_HR10{hr10:.4f}_Loss{loss.item():.4f}.pkl"
                    )
                    dirpath = os.path.dirname(fn)
                    os.makedirs(dirpath, exist_ok=True)

                    torch.save(net.state_dict(), fn)
                    
                    if hr10 > self.best_hr10:
                        self.best_hr10 = hr10
                        self.best_epoch = epoch
                        self.no_improvement_count = 0  
                        os.makedirs(f"./model/{self.dataset}", exist_ok=True)
                        bestfn = (
                            f"./model/{self.dataset}/best.pkl"
                        )
                        torch.save(net.state_dict(), bestfn)
                    else:
                        self.no_improvement_count += 2 
                    if self.no_improvement_count >= self.early_stop:
                        print("Early stopping.")
                        print("Average training time per epoch: {:.4f}s".format(total_train_time / (epoch + 1)))
                        print("Average prediction time per evaluation: {:.4f}s".format(total_predict_time / ((epoch // 2) + 1)))
                        return bestfn
        print("Training complete.")
        print("Average training time per epoch: {:.4f}s".format(total_train_time / (epoch + 1)))
        print("Average prediction time per evaluation: {:.4f}s".format(total_predict_time / ((epoch // 2) + 1)))
        return bestfn
