import numpy as np
import torch
import pytorch_lightning as pl
import time
import copy

import data as my_data
from lightning.lightning import LightningRNNOneHot
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import MLFlowLogger

from models.gated_recurrent_unit import GatedRecurrentUnit
from models.long_short_term_memory import LongShortTermMemory




_MODELS_ARH_DICT_ = {
    "GatedRecurrentUnit": GatedRecurrentUnit,
    "LongShortTermMemory": LongShortTermMemory
}

_MODELS_PYTORCH_LIGHTNING_DICT_ = {
    "LightningRNNOneHot": LightningRNNOneHot
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
    encoder, total_samples = my_data.load('./data/*.txt')

    train_samples, val_samples = np.split(total_samples,
                                            [int(.9 * len(total_samples))])
    print("Total samples:{} = train:{}, valid:{}".format(
        len(total_samples), len(train_samples), len(val_samples)))
    del total_samples

    n_categories = len(encoder.all_categories)
    input_size = len(encoder.all_letters)
    output_size = len(encoder.all_letters)

    train_dl = DataLoader(my_data.CityNamesOneHot(encoder, train_samples, device="cuda"),
                            shuffle=True,
                            num_workers=6,
                            batch_size=train_config["batch_size"],
                            collate_fn=my_data.pad_collate,
                            drop_last=True,
                            persistent_workers=True)
    val_dl = DataLoader(my_data.CityNamesOneHot(encoder, val_samples, device="cuda"),
                            shuffle=False,
                            batch_size=1,
                            collate_fn=my_data.pad_collate)
    train_config["train_elements"] = len(train_samples)
    train_config["test_elements"] = len(val_samples)
    
    
    # ----------------------- Подготовка модели -----------------------
    model_arh = MODEL_ARH(**train_config["model_architecture"]["args"])
    
    model = MODEL_PYTORCH_LIGHTNING(model_arh, 
                                    **train_config["model_pytorch_lightning"]["args"])
    
    logger = MLFlowLogger(experiment_name=f"RNN-train_model", 
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
            dirpath=f'.\\checkpoints\\checkpoint_{start_time_train}',
            filename=r"bin_class\rnn-{epoch:03d}-{val_loss:.3f}", 
            monitor="val_loss",
            save_top_k=5,
            save_last=True,
            mode="min",
    )]

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=150, 
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
    
if __name__ == "__main__":
    train_config = {
        "model_architecture": {"type": "LongShortTermMemory",
                               "args": {"input_size": 166, "hidden_size": 166}},
        "model_pytorch_lightning": {"type": "LightningRNNOneHot",
                                    "args": {"learning_rate": 1e-3, 
                                             "padding_index": my_data.PAD_ID,
                                             "teacher_forcing": 0.5,
                                             }},
        "batch_size": 16,
        "max_epochs": 50,
    }
    test_config = {
        "model_pytorch_lightning/args/teacher_forcing": [0.3, 0.5, 0.8, 1],
        # "batch_size": [8, 16, 32],
    }
    
    for conf in parameters_generator(test_config):
        new_conf = copy.deepcopy(train_config)
        for key in conf.keys():
            set_on_dict_path(new_conf, key.split("/"), conf[key])
        
        print("Start training with config: " + str(new_conf))
        main(new_conf)
        time.sleep(10)
        
       