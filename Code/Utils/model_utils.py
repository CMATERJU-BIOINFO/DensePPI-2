import numpy as np
import copy, os, torch

class ModelCheckpoint:
    def __init__(self, model_save_load_dir : str = "/some/dir/", model_save_prefix : str = "complete", mode = 'max'):
        self.mode = mode
        self.model_dir = model_save_load_dir
        self.model_save_prefix = model_save_prefix
        self.num_epochs = 0
        self.metric_arr = []
        self.saved_at = ""
        self.model_state_dict_key = 'model_state_dict'
        self.optimizer_state_dict_key = 'optimizer_state_dict'
        self.metric_dict_key = 'metric'
        self.num_epochs_key = 'num_epochs'
        if(mode == 'max'):
            self.metric = -np.inf
        if(mode == 'min'):
            self.metric = np.inf
        self.model_state = None
        self.optim_state = None
        os.makedirs(self.model_dir, exist_ok=True)

    def get_metric_arr(self):
        return self.metric_arr
    
    def get_last_saved_loc(self):
        return self.saved_at
    
    def get_num_epochs(self):
        return self.num_epochs
    
    def load_checkpoint(self, model_load_file_name : str = "some.pth.tar", model = None, optim = None):
        self.saved_at = os.path.join(self.model_dir, model_load_file_name)
        checkpoint = torch.load(self.saved_at)
        self.num_epochs = checkpoint[self.num_epochs_key]
        self.metric = checkpoint[self.metric_dict_key]
        model.load_state_dict(checkpoint[self.model_state_dict_key])
        self.model_state = copy.deepcopy(model)
        if (optim != None):
            optim.load_state_dict(checkpoint[self.optimizer_state_dict_key])
            self.optim_state = copy.deepcopy(optim)
        
        print(f"Model loaded successfully from checkpoint with metric = {self.metric:.5f}, num_epochs = {self.num_epochs:03d}...")

    def save_checkpoint(self):
        print("=> Saving a new best")
        save_file_name = f'{self.model_save_prefix}_epoch_{self.num_epochs:03d}_metric_{self.metric:.5f}.pth.tar'
        save_at = os.path.join(self.model_dir, save_file_name)
        if (self.optim_state != None):
            state_dict = {self.num_epochs_key : self.num_epochs,
                        self.model_state_dict_key : self.model_state.state_dict(),
                        self.optimizer_state_dict_key : self.optim_state.state_dict(),
                        self.metric_dict_key : self.metric}
        else:
            state_dict = {self.num_epochs_key : self.num_epochs,
                        self.model_state_dict_key : self.model_state.state_dict(),
                        self.metric_dict_key : self.metric}
        torch.save(state_dict, save_at)
        self.saved_at = save_at
        print("Best model saved at location : ", save_at)
        

    def change_current(self, is_best, metric, state, optim):
        if is_best:
            self.metric = metric
            self.model_state = copy.deepcopy(state)
            if optim != None :
                self.optim_state = copy.deepcopy(optim)
            self.save_checkpoint()
        else:
            print("=> Metric did not improve")

    def create_checkpoint(self, metric, state, optim = None):
        self.num_epochs += 1
        self.metric_arr.append(self.metric)
        if(self.mode == 'max'):
            is_best = bool(metric > self.metric)
            self.change_current(is_best, metric, state, optim)
        
        if(self.mode == 'min'):
            is_best = bool(metric < self.metric)
            self.change_current(is_best, metric, state, optim)