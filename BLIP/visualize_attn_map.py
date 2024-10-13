import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os


attn_wt_path = "./experiments/visualize/l2-global-smoothing-sharedCodebooks-layer11-3AlignBlocks_frozen-update-x-cross-SwapAttnInps/TEXT_[CLS]__harry__g__frankfurt__,__a__man__with__a__beard__and__mustache__,__is__sitting__at__a__table__with__a__microphone__in__front__of__him__he__is__wearing__a__suit__and__tie__and__[SEP]_layer0_cross.pt"

# attn_wt_path = "experiments/visualize/l2-global-smoothing-sharedCodebooks-layer11-3AlignBlocks_frozen_residual/TEXT_a_train_station_with_a_train_on_the_tracks_and_a_pedestrian_crossing_layer0_cross.pt"

for num in range(8):
    head_no = num
    attn_wts = torch.load(attn_wt_path)[0][head_no]
    print("attn_wts", attn_wts.size())


    a, b = attn_wts.size(0), attn_wts.size(1)
    scale = 1.0/3
    a, b = int(a*scale), int(b*scale)

    print(a, b)
    # Plot token-level similarity heatmaps
    plt.figure(figsize=(b, a))
    sns.heatmap(attn_wts.detach().cpu().numpy(), annot=True, cmap="rocket_r", fmt=".2f", cbar=False, vmin=0, vmax=1)
    token_names = attn_wt_path.split("__")[1:]
    print(token_names)
    plt.xticks(torch.arange(len(token_names)), token_names, rotation=90, ha='center')
    # plt.yticks(np.arange(B*T) + 0.5, token_names)
    # plt.xlabel('Token')
    # plt.ylabel('Token')
    plt.tick_params(axis='x', labelbottom=False, bottom=False, top=True, labeltop=True)
    plt.title(f'Attn Wts')
    plt.tight_layout()

    save_path = f"./token_emb_plots/{'/'.join(attn_wt_path.split('/')[-2:])[:-3]}_head{head_no}.png"
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()

    print(f"Saved at {save_path}")