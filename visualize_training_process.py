from argparse import ArgumentParser
import matplotlib.pyplot as plt
from os.path import abspath, join
import pandas as pd
from icecream import ic

if __name__ == "__main__":
    parser = ArgumentParser(prog='plot training graphs')
    
    parser.add_argument('-r', '--trainings-run', required=True, type=str)
    parser.add_argument('-o', '--overlap-graphs', required=False, type=str, nargs="*", default=[])

    args = parser.parse_args()

    print(args.overlap_graphs)
        
    df = pd.read_csv(abspath(join('results', 'train', args.trainings_run, "data.csv")))
    plt.figure()

    max_IoU = max(df['val_IoU'])
    ic(df['val_IoU'] == max_IoU)
    max_epoch = (df['val_IoU'] == max_IoU).loc[(df['val_IoU'] == max_IoU) == True].index.values[0]                                          
    max_epoch_loss = df.iloc[max_epoch]['val_loss']
    
    for i in args.overlap_graphs:
        ipd = pd.read_csv(abspath(join('results', 'train', args.trainings_run, i, "data.csv")))

        true_df = df == ipd.iloc[0]
        start_index = true_df.loc[true_df["train_IoU"] == True].index
        ic(start_index)

        
        plt.plot([i + start_index for i in ipd.index], ipd['train_IoU'], color="gray")
        plt.plot([i + start_index for i in ipd.index], ipd['train_acc'], color="gray")
        plt.plot([i + start_index for i in ipd.index], ipd['train_loss'], color="gray")
        plt.plot([i + start_index for i in ipd.index], ipd['val_IoU'], color="gray", linestyle='--')
        plt.plot([i + start_index for i in ipd.index], ipd['val_acc'], color="gray", linestyle='--')
        plt.plot([i + start_index for i in ipd.index], ipd['val_loss'], color="gray", linestyle='--')

        i_max_IoU = max(ipd['val_IoU'])
        i_max_epoch = ipd.iloc
        if i_max_IoU > max_IoU:
            max_IoU = i_max_IoU


    plt.plot(df.index, df['train_IoU'], color="blue", label="Train IoU")
    plt.plot(df.index, df['train_acc'], color="cyan", label="Train Acc")
    plt.plot(df.index, df['train_loss'], color="green", label="TrainLoss")
    plt.plot(df.index, df['val_IoU'], color="blue", linestyle='--', label="Val IoU")
    plt.plot(df.index, df['val_acc'], color="cyan", linestyle='--', label="Val Acc")
    plt.plot(df.index, df['val_loss'], color="green", linestyle='--', label="Val Loss")

    plt.legend()

    output_file = abspath(join('results', 'train', args.trainings_run, "data.svg" if args.overlap_graphs == [] else "data_all.svg"))
    plt.savefig(output_file)

    print(f"max IoU at epoch {max_epoch}: -IoU: {max_IoU} -Loss: {max_epoch_loss} ({output_file})")