import numpy as np
import torch
import pytorch_lightning as pl

import data as my_data
from models.gated_recurrent_unit import GatedRecurrentUnit
from lightning.lightning import LightningRNNOneHot
from torch.utils.data import DataLoader

def main():
    SEED = 2334
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    MODULE = LightningRNNOneHot

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
                            batch_size=16,
                            collate_fn=my_data.pad_collate,
                            drop_last=True,
                            persistent_workers=True)
    val_dl = DataLoader(my_data.CityNamesOneHot(encoder, val_samples, device="cuda"),
                            shuffle=False,
                            batch_size=1,
                            collate_fn=my_data.pad_collate)


    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            filename="rnn-{epoch:03d}-{val_loss:.2f}",
    )]
    # train_samples = my_data.train_samples(encoder)
    # val_samples = my_data.val_samples(encoder)
    model = MODULE(GatedRecurrentUnit(input_size, output_size),
                   learning_rate=1e-3)
    trainer = pl.Trainer(max_epochs=150, accelerator="gpu", callbacks=callbacks)
    trainer.fit(model, 
                train_dataloaders=train_dl, 
                val_dataloaders=val_dl)
    
if __name__ == "__main__":
    main()