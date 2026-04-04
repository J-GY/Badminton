from Trainer import TRIDENT_Trainer
import torch

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    trident = TRIDENT_Trainer()

    load_model_name = None
    load_optimizer_name = None 

    best_model_path = trident.TRIDENT_train(load_model=load_model_name, load_optimizer=load_optimizer_name)
    trident.TRIDENT_eval(load_model=best_model_path)
