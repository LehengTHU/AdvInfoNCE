#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# %%
# file_dir = "weights/coat/AdvDRO-LGN/4_28_AdvDRO_mlp_pknm_coat_tau_0.75_n_layers_2_lr_0.0005_i_10_ae_1_al_0.005_w_0_eta_70_batch_1024_neg_64_patience_30_k_neg_64.0stats_.txt"
# file_dir = "weights/coat/AdvDRO-LGN/4_28_AdvDRO_mlp_pknm_coat_tau_0.75_n_layers_2_lr_0.0005_i_10_ae_1_al_0.005_w_0_eta_35_batch_1024_neg_64_patience_30_k_neg_64.0stats_.txt"
# file_dir = "weights/tencent.new/AdvDRO/2_20_AdvDRO_i_5_ae_2_al_5e_5_eta_6_n_layers=0tau=0.1stats_.txt"
# file_dir = "weights/yahoo.new/AdvDRO/4_26_sota_AdvDRO_mlp_pknm_yahoo.new_tau_0.28_n_layers_0_lr_0.0005_i_15_ae_1_al_0.0001_w_10_eta_15_batch_1024_neg_128_patience_10_k_neg_64.0stats_.txt"
# %%
def read_log(file_dir, show=False):
    # 逐行读取file_dir文件, 只保留
    id_recall = []
    id_ndcg = []
    ood_recall = []
    ood_ndcg = []

    with open(file_dir, 'r') as f:
        # count = 0
        for line in f:
            line = line.split(' ')
            if("valid" in line[0]):
                id_recall.append(float(line[1][:-1]))
                id_ndcg.append(float(line[7][:-1]))
            if("test_ood" in line[0]):
                ood_recall.append(float(line[1][:-1]))
                ood_ndcg.append(float(line[7][:-1]))

    epochs = list(range(0, len(id_recall)))
    epochs = [i*5 for i in epochs]
    # 定义表格
    result = pd.DataFrame({'epochs': epochs, 'id_recall': id_recall, 'ood_recall': ood_recall, 'id_ndcg': id_ndcg, 'ood_ndcg': ood_ndcg})
    # df是除了最后一行的所有行
    df = result.iloc[:-1, :]

    fig=plt.figure()
    x = df.epochs
    y1 = df.id_recall
    y2 = df.ood_recall
    print(max(y1), max(y2), 1.1*max(y1), 1.1*max(y2))
    #ax1显示y1  ,ax2显示y2 
    ax1=fig.subplots()
    ax2=ax1.twinx()    #使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
    ax1.plot(x,y1,'g-', label='id_recall')
    ax2.plot(x,y2,'b--', label='ood_recall')
    # 坐标轴范围
    ax1.set_ylim(min(y1), 1.15*(max(y1)-min(y1))+min(y1))
    ax2.set_ylim(min(y2), 1.15*(max(y2)-min(y2))+min(y2))

    ax1.set_xlabel('epochs')
    ax1.set_ylabel('id_recall')
    ax2.set_ylabel('ood_recall')
    # legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # ax1.legend(['id_recall'], loc='upper left')
    # ax2.legend(['ood_recall'], loc='upper right')

    # base_path = "content/log_result/" + file_dir.split("/")[-1][:-5]
    # if not os.path.exists(base_path):
        # os.makedirs(base_path)
    base_path = file_dir[:-10]
    save_path = base_path + "/train_log.png"
    plt.savefig(save_path)
    if(show):
        plt.show()
    save_path = base_path + "/train_log.csv"
    result.to_csv(save_path, index=False)

# %%
# file_dir = "weights/yahoo.new/AdvDRO-LGN/4_30_r_ns_64_sota_r_AdvDRO_mlp_pknm_yahoo.new_tau_0.28_n_layers_2_lr_0.0005_i_10_ae_1_al_0.0001_w_0_eta_15_batch_1024_neg_64_patience_10_k_neg_64.0stats_.txt"
# file_dir = "weights/kuairec_ood/AdvDRO-LGN/4_29_v1_AdvDRO_embed_pknm_kuairec_ood_tau_0.5_n_layers_2_lr_1e-06_i_2_ae_1_al_5e-05_w_0_eta_40_batch_2048_neg_64_patience_10_k_neg_32.0stats_.txt"
# read_log(file_dir, True)
# %%
# from reckit import randint_choice
# import numpy as np
# %%
# numpy生成一个2231长度的随机序列
# np.random.seed(0)
# a = np.random.randint(0, 10728, 2231)
# s = randint_choice(10728, size=128, exclusion=a)
# %%
