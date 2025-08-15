import torch



class GatedRecurrentUnit(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # http://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.last_hidden = None

    def forward(self, x, hidden = None) -> torch.Tensor:
        # if hidden is None and self.last_hidden is None:
        #     hidden = self.init_hidden(self.hidden_size)
        # elif hidden is None and self.last_hidden is not None:
        #     hidden = self.last_hidden
        prediction, self.last_hidden = self.gru.forward(x, hidden)
        return prediction, self.last_hidden
    
    def init_hidden(self, batch_size = 1, batched = False, device = torch.device('cpu')):
        """
        Возвращает начальное скрытое состояние GRU

        :param batch_size: размер батча
        :param batched: признак батчирования

        :return: начальное скрытое состояние (1, output_size) или (1, batch_size, output_size) при батчировании

        """
        if batched:
            return torch.zeros(1, batch_size, self.gru.hidden_size).to(device)
        else:
            return torch.zeros(1, self.gru.hidden_size).to(device)
        
    def __str__(self):
        return "GatedRecurrentUnit(input_size={}, hidden_size={})".format(self.input_size, self.hidden_size)



        