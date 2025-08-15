from torch.nn import functional as F
import pytorch_lightning as pl
# from ..models.gated_recurrent_unit import GatedRecurrentUnit
import torch
import matplotlib.pyplot as plt
import matplotlib


class LightningRNNOneHot(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, padding_index=0, teacher_forcing: float = 0 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = learning_rate
        self.padding_index = padding_index
        self.teacher_forcing = teacher_forcing
        

    def training_step(self, batch, batch_idx):
        c, x, y = batch
        loss = torch.tensor(0.0).to(x.device)
        hidden = None
        for i in range(x.shape[1]):
            if hidden is None:
                hidden = self.model.init_hidden(y.shape[0], batched=True, device=self.device)
            else:    
                try_hidden = torch.index_select(x, 1, torch.tensor(i-1).to(self.device)).permute(1, 0, 2) * self.teacher_forcing + \
                    (hidden[1] if isinstance(hidden, tuple) else hidden) * (1 - self.teacher_forcing)
                if isinstance(hidden, tuple):
                    hidden = (try_hidden, hidden[1])
                else:
                    hidden = try_hidden

            x_select = torch.index_select(x, 1, torch.tensor(i).to(self.device)) # [batch_size, 1, input_size]
            y_select = torch.index_select(y, 1, torch.tensor(i).to(self.device)) # [batch_size, 1, input_size]
            prediction, hidden = self.model.forward(x_select, hidden)
            mask = y_select.ravel() != self.padding_index # remove padding
            loss += F.cross_entropy(prediction[:, 0][mask], y_select.view(-1)[mask])

        loss /= len(x)    


        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        c, x, y = batch
        loss = torch.tensor(0.0).to(x.device)
        hidden = None
        for i in range(x.shape[1]):
            if hidden is None:
                hidden = self.model.init_hidden(y.shape[0], batched=True, device=self.device)
            else:    
                try_hidden = torch.index_select(x, 1, torch.tensor(i-1).to(self.device)).permute(1, 0, 2)
                if isinstance(hidden, tuple):
                    hidden = (try_hidden, hidden[1])
                else:
                    hidden = try_hidden

            x_select = torch.index_select(x, 1, torch.tensor(i).to(self.device))
            y_select = torch.index_select(y, 1, torch.tensor(i).to(self.device))
            prediction, hidden = self.model.forward(x_select, hidden)
            mask = y_select.ravel() != self.padding_index # remove padding
            loss += F.cross_entropy(prediction[:, 0][mask], y_select.view(-1)[mask])

        loss /= len(x)    

        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer