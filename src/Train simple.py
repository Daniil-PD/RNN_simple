import numpy as np
import torch
import pytorch_lightning as pl
import time
import copy

import data as my_data
from lightning.LightningRNNOneHot import LightningRNNOneHot
from lightning.LightningRNNTimeSeries import LightningRNNTimeSeries_recursive
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import MLFlowLogger

from models.gated_recurrent_unit import GatedRecurrentUnit
from models.long_short_term_memory import LongShortTermMemory




_MODELS_ARH_DICT_ = {
    "GatedRecurrentUnit": GatedRecurrentUnit,
    "LongShortTermMemory": LongShortTermMemory
}

_MODELS_PYTORCH_LIGHTNING_DICT_ = {
    "LightningRNNOneHot": LightningRNNOneHot,
    "LightningRNNTimeSeries": LightningRNNTimeSeries_recursive
}

def main(train_config):
    SEED = 2334
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    
    start_time_train = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    MODEL_ARH = _MODELS_ARH_DICT_[train_config["model_architecture"]["type"]]
    MODEL_PYTORCH_LIGHTNING = _MODELS_PYTORCH_LIGHTNING_DICT_[train_config["model_pytorch_lightning"]["type"]]

    # ----------------------- Подготовка данных -----------------------
    with torch.no_grad():
        y = func(torch.arange(0, 300, 0.1))
        total_samples = get_samples(y, train_config["window_size"], train_config["out_size"])

        train_samples = total_samples[:int(.9 * len(total_samples))]
        val_samples = total_samples[int(.9 * len(total_samples)):]
        print("Total samples:{} = train:{}, valid:{}".format(
            len(total_samples), len(train_samples), len(val_samples)))
        del total_samples


    train_dl = DataLoader(train_samples,
                            shuffle=True,
                            num_workers=6,
                            batch_size=train_config["batch_size"],
                            drop_last=True,
                            persistent_workers=True)
    val_dl = DataLoader(val_samples,
                            shuffle=False,
                            batch_size=1)
    train_config["train_elements"] = len(train_samples)
    train_config["test_elements"] = len(val_samples)
        
    
    
    # ----------------------- Подготовка модели -----------------------
    model_arh = MODEL_ARH(**train_config["model_architecture"]["args"])
    
    model = MODEL_PYTORCH_LIGHTNING(model_arh, 
                                    **train_config["model_pytorch_lightning"]["args"])
    
    logger = MLFlowLogger(experiment_name=f"RNN-train_model_time_series", 
                      tags={"used_architecture": model.model.__class__.__name__}, 
                      tracking_uri="http://127.0.0.1:5000",
                      log_model=True,
                      run_name=start_time_train + "_" + train_config["model_architecture"]["type"] 
                    #   artifact_location=f'.\\checkpoints\\checkpoint_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'
                      )
    logger.log_hyperparams(train_config)

    # ----------------------- Подготовка обучения -----------------------
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f'.\\checkpoints_time_series\\checkpoint_{start_time_train}',
            filename=r"bin_class\rnn-{epoch:03d}-{val_loss:.3f}", 
            monitor="val_loss",
            save_top_k=3,
            save_last=True,
            mode="min",
    )]

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=train_config["max_epochs"], 
        accelerator="gpu", 
        callbacks=callbacks,
        log_every_n_steps=25,
        check_val_every_n_epoch=1)
    
    # ----------------------- Обучение модели -----------------------
    
    trainer.fit(model, 
                train_dataloaders=train_dl, 
                val_dataloaders=val_dl)
    
     
def parameters_generator(parameters_ranges: dict):
    params_selector = {key: 0 for key in parameters_ranges.keys()}

    while True:
        parameters = {}
        for key, value in parameters_ranges.items():
            parameters[key] = value[params_selector[key]]

        yield parameters
        
        for key in params_selector.keys():
            if params_selector[key] < len(parameters_ranges[key]) - 1:
                params_selector[key] += 1
                break
            else:
                params_selector[key] = 0
        else:
            return

def set_on_dict_path(in_dict, path: list, value):
    if len(path) == 1:
        in_dict[path[0]] = value
        return
    set_on_dict_path(in_dict[path[0]], path[1:], value)
    
def func(x):
    y = np.sin(x)+np.sin(x*2-2.6)+np.sin(x*3)+np.sin(x*4-1.5)+np.random.normal(0, 0.1, len(x)).astype(np.float32)
    return y

def get_samples(y, window_size, outsize):
    return [(y[i:i+window_size].view(window_size, 1), y[i+window_size:i+window_size+outsize].view(outsize, 1)) for i in range(len(y)-window_size-outsize)]

if __name__ == "__main__":
    train_config = {
        "model_architecture": {"type": "LongShortTermMemory", # "GatedRecurrentUnit", "LongShortTermMemory"
                               "args": {"input_size": 1, "hidden_size": 1}},
        "model_pytorch_lightning": {"type": "LightningRNNTimeSeries",
                                    "args": {"learning_rate": 1e-3, 
                                             "teacher_forcing": 0.1,
                                             }},
        "batch_size": 32,
        "max_epochs": 150,
        "window_size": 75,
        "out_size": 25 
    }
    test_config = {
        # "model_architecture/type": ["GatedRecurrentUnit", "LongShortTermMemory"],
        # "model_pytorch_lightning/args/teacher_forcing": [0, 0.5, 0.8, 1],
        # "batch_size": [16, 32, 64],
        # "model_pytorch_lightning/args/learning_rate": [1e-4, 1e-5],
    }
    
    for conf in parameters_generator(test_config):
        new_conf = copy.deepcopy(train_config)
        for key in conf.keys():
            set_on_dict_path(new_conf, key.split("/"), conf[key])
        
        print("Start training with config: " + str(new_conf))
        main(new_conf)
        time.sleep(10)
        
       