import data as my_data
import numpy as np
import torch
import glob


from models.gated_recurrent_unit import GatedRecurrentUnit
from lightning.lightning import LightningRNNOneHot



def main():




    encoder, total_samples = my_data.load('data/*.txt')
    train_samples, val_samples = np.split(total_samples,
                                          [int(.9 * len(total_samples))])
    # print("Total samples:{} = train:{}, valid:{}".format(
    #     len(total_samples), len(train_samples), len(val_samples)))
    del total_samples

    n_categories = len(encoder.all_categories)
    input_size = len(encoder.all_letters)
    output_size = len(encoder.all_letters)


    MODULE = LightningRNNOneHot
    model = MODULE(GatedRecurrentUnit(input_size, output_size))

    ckpt_path = glob.glob("./lightning_logs/*/checkpoints/*.ckpt")[-1]
    print("Loading model from {}".format(ckpt_path))
    # ckpt_path = r"lightning_logs\version_22\checkpoints\epoch=9-step=3470.ckpt"
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict)

    pretrained_model = model
    
    # predict
    pretrained_model.eval()
    pretrained_model.freeze()


    # print(encoder.decode(next(iter(pretrained_model.val_dataloader()))[1]))
    my_data.generate(pretrained_model.model, encoder, 'ru', start_chars=[my_data.BOS]*20, max_length=20, temperature=0.2)

    # while True:
    #     try:
    #         inp = input("Enter a string: ")
    #         x = encoder.encode(inp)
    #         print(x.size())
    #         y_hat = pretrained_model.forward(category, x)
    #         print(y_hat)
    #     except KeyboardInterrupt as e:
    #         break
    # return

if __name__ == '__main__':
    main()