from torch.nn import functional as F
import pytorch_lightning as pl
# from ..models.gated_recurrent_unit import GatedRecurrentUnit
import torch
import matplotlib.pyplot as plt
import matplotlib


class LightningRNNTimeSeries_recursive(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, teacher_forcing: float = 0 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = learning_rate
        self.teacher_forcing = teacher_forcing
        

    def training_step(self, batch, batch_idx):
        x, y = batch        
        loss = torch.tensor(0.0).to(x.device)
        
        # Предсказание
        hidden = None
        
        for i in range(y.shape[1]):
            if hidden is None:
                # Прогрев на данных
                hidden = self.model.init_hidden(y.shape[0], batched=True, device=self.device)
                prediction, hidden = self.model.forward(x, hidden)
            else:
                try_hidden = torch.index_select(y, 1, torch.tensor(i-1).to(self.device)).permute(1, 0, 2) * self.teacher_forcing + \
                    (hidden[1] if isinstance(hidden, tuple) else hidden) * (1 - self.teacher_forcing)
                if isinstance(hidden, tuple):
                    hidden = (try_hidden, hidden[1])
                else:
                    hidden = try_hidden

                prediction, hidden = self.model.forward(y_select, hidden)

            y_select = torch.index_select(y, 1, torch.tensor(i).to(self.device)) # [batch_size, 1, input_size]            
            loss += F.mse_loss(prediction[:, 0], y_select[:, 0])

        loss /= len(y)  

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.tensor(0.0).to(x.device)
        
        # Предсказание
        hidden = None
        
        for i in range(y.shape[1]):
            if hidden is None:
                # Прогрев на данных
                hidden = self.model.init_hidden(y.shape[0], batched=True, device=self.device)
                prediction, hidden = self.model.forward(x, hidden)
            else:
                try_hidden = torch.index_select(y, 1, torch.tensor(i-1).to(self.device)).permute(1, 0, 2) * self.teacher_forcing + \
                    (hidden[1] if isinstance(hidden, tuple) else hidden) * (1 - self.teacher_forcing)
                if isinstance(hidden, tuple):
                    hidden = (try_hidden, hidden[1])
                else:
                    hidden = try_hidden

                prediction, hidden = self.model.forward(y_select, hidden)

            y_select = torch.index_select(y, 1, torch.tensor(i).to(self.device)) # [batch_size, 1, input_size]            
            loss += F.mse_loss(prediction[:, 0], y_select[:, 0])

        loss /= len(y)    

        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer