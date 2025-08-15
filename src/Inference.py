import data as my_data
import numpy as np
import torch
import glob


from models.gated_recurrent_unit import GatedRecurrentUnit
from models.long_short_term_memory import LongShortTermMemory
from lightning.LightningRNNOneHot import LightningRNNOneHot



def main():

    encoder, total_samples = my_data.load('data/names/*.txt')
    train_samples, val_samples = np.split(total_samples,
                                          [int(.9 * len(total_samples))])
    # print("Total samples:{} = train:{}, valid:{}".format(
    #     len(total_samples), len(train_samples), len(val_samples)))
    del total_samples

    n_categories = len(encoder.all_categories)
    input_size = len(encoder.all_letters)
    output_size = len(encoder.all_letters)


    MODULE = LightningRNNOneHot
    

    # ckpt_path = glob.glob("./lightning_logs/*/checkpoints/*.ckpt")[-1]
    ckpt_path = r"checkpoints\checkpoint_2025-08-01_18-48-51\last.ckpt"
    # ckpt_path = r"checkpoints\checkpoint_2025-08-01_18-05-21\last.ckpt"
    # ckpt_path = r"checkpoints\checkpoint_2025-08-02_06-59-51\last.ckpt"
    ckpt_path = r"checkpoints\checkpoint_2025-08-04_18-34-23\bin_class\rnn-epoch=074-val_loss=1240.830.ckpt"
    
    
    print("Loading model from {}".format(ckpt_path))
    
    state_dict = torch.load(ckpt_path)["state_dict"]
    print(state_dict.keys())
    try:
        model = MODULE(LongShortTermMemory(input_size, output_size))
        model.load_state_dict(state_dict)
        print("Loaded LongShortTermMemory")
    except:
        model = MODULE(GatedRecurrentUnit(input_size, output_size))
        model.load_state_dict(state_dict)
        print("Loaded GatedRecurrentUnit")

    pretrained_model = model
    
    # predict
    pretrained_model.eval()
    pretrained_model.freeze()


    # print(encoder.decode(next(iter(pretrained_model.val_dataloader()))[1]))
    my_data.generate(pretrained_model.model, encoder, 'ru', start_chars=[my_data.BOS]*10, max_length=300, temperature=0.2)

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