import random
import re
from sys import get_coroutine_origin_tracking_depth
from sys import exit
random.seed(101)
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
#from scipy.linalg import svd
import itertools
import torch
import time
import numpy as np
from tqdm import tqdm
from evaluator import ProxyEvaluator
import collections
import os
from data import Data
from parse import parse_args
# need change 1
from model import CausE, IPS, LGN, MACR, INFONCE, INFONCE_linear, AdvDRO, SAMREG, BC_LOSS, SimpleX, SGL, NCL, XSimGCL, PID, NBC_LOSS
from torch.utils.data import Dataset, DataLoader
from collect_log import read_log
import torch.nn.functional as F


def merge_user_list(user_lists):
    out = collections.defaultdict(list)
    # 循环遍历每个用户列表
    for user_list in user_lists:
        # 循环遍历每个用户
        for key, item in user_list.items():
            out[key] = out[key] + item
    return out


def merge_user_list_no_dup(user_lists):
    out = collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key] = out[key] + item
    
    for key in out.keys():
        out[key]=list(set(out[key]))
    return out


def save_checkpoint(model, epoch, checkpoint_dir, buffer, max_to_keep=10):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)
    buffer.append(filename)
    if len(buffer)>max_to_keep:
        os.remove(buffer[0])
        del(buffer[0])

    return buffer


def restore_checkpoint(model, checkpoint_dir, device, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0,

    epoch_list = []

    regex = re.compile(r'\d+')

    for cp in cp_files:
        epoch_list.append([int(x) for x in regex.findall(cp)][0])

    epoch = max(epoch_list)

   
    if not force:
        print("Which epoch to load from? Choose in range [0, {})."
              .format(epoch), "Enter 0 to train from scratch.")
        print(">> ", end = '')
        # inp_epoch = int(input())
        inp_epoch = epoch
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0,
    else:
        print("Which epoch to load from? Choose in range [0, {}).".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(0, epoch):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename, map_location = str(device))

    try:
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
              .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch


def restore_best_checkpoint(epoch, model, checkpoint_dir, device):
    """
    Restore the best performance checkpoint
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename, map_location = str(device))

    model.load_state_dict(checkpoint['state_dict'])
    print("=> Successfully restored checkpoint (trained for {} epochs)"
          .format(checkpoint['epoch']))

    return model


def clear_checkpoint(checkpoint_dir):
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def evaluation(args, data, model, epoch, base_path, evaluator, name="valid"):
    # Evaluate with given evaluator

    ret, _ = evaluator.evaluate(model)

    n_ret = {"recall": ret[1], "hit_ratio": ret[5], "precision": ret[0], "ndcg": ret[3], "mrr":ret[4], "map":ret[2]}

    perf_str = name+':{}'.format(n_ret)
    print(perf_str)
    with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(perf_str + "\n")
    # Check if need to early stop (on validation)
    is_best=False
    early_stop=False
    if name=="valid":
        if ret[1] > data.best_valid_recall:
            data.best_valid_epoch = epoch
            data.best_valid_recall = ret[1]
            data.patience = 0
            is_best=True
        else:
            data.patience += 1
            if data.patience >= args.patience:
                print_str = "The best performance epoch is % d " % data.best_valid_epoch
                print(print_str)
                early_stop=True

    return is_best, early_stop


def Item_pop(args, data, model):

    for K in range(5):

        eval_pop = ProxyEvaluator(data, data.train_user_list, data.pop_dict_list[K], top_k=[(K+1)*10],
                                   dump_dict=merge_user_list([data.train_user_list, data.valid_user_list]))

        ret, _ = eval_pop.evaluate(model)

        print_str = "Overlap for K = % d is % f" % ( (K+1)*10, ret[1] )

        print(print_str)

        with open('stats_{}.txt'.format(args.saveID), 'a') as f:
            f.write(print_str + "\n")


def ensureDir(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def split_grp_view(data,grp_idx):
    n=len(grp_view)
    split_data=[{} for _ in range(n)]

    for key,item in data.items():
        for it in item:
            if key not in split_data[grp_idx[it]].keys():
                split_data[grp_idx[it]][key]=[]
            split_data[grp_idx[it]][key].append(it)
    return split_data


def checktensor(tensor):
    t=tensor.detach().cpu().numpy()
    if np.max(np.isnan(t)):        
        idx=np.argmax(np.isnan(t))
        return idx
    else:
        return -1

def get_rotation_matrix(axis, theta):
    """
    Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


grads = {}
def save_grad(name):
    def hook(grad):
        torch.clamp(grad, -1, 1)
        grads[name] = grad
    return hook

"""
It takes in the embeddings of users and items, the popularity of users and items, and the data, and
plots the embeddings of users and items in 3D space, and plots the angular distribution of the
embeddings of items. 

The embeddings of users and items are plotted in 3D space, and the angular distribution of the
embeddings of items are plotted in 2D space. 

:param items: the item embeddings
:param users: the user embeddings
:param data: the data object
:param p_item: the popularity of each item
:param p_user: the popularity of each user
:param name: the name of the dataset
"""
def visulization(items,users,data,p_item,p_user,name):
    test_ood_user_list=data.test_ood_user_list
    test_id_user_list=data.test_id_user_list
    train_user_list=data.train_user_list

    def split_grp_view(data,grp_idx):
        n=len(grp_view)
        split_data=[collections.defaultdict(list) for _ in range(n)]

        for key,item in data.items():
            for it in item:
                if key not in split_data[grp_idx[it]].keys():
                    split_data[grp_idx[it]][key]=[]
                split_data[grp_idx[it]][key].append(it)
        return split_data

    pop_sorted=np.sort(p_item)
    n_items=p_item.shape[0]

    n_groups=3
    grp_view=[]
    for grp in range(n_groups):
        split=int((n_items-1)*(grp+1)/n_groups)
        grp_view.append(pop_sorted[split])
    #print("group_view:",grp_view)
    idx=np.searchsorted(grp_view,p_item)

    pop_group=[[] for _ in range(n_groups)]
    for i in range(n_items):
        pop_group[idx[i]].append(i)

    eval_test_ood_split=split_grp_view(test_ood_user_list,idx)
    eval_test_id_split=split_grp_view(test_id_user_list,idx)
    eval_train_split=split_grp_view(train_user_list,idx)

    pop_users=p_user.tolist()

    u_pop_sorted=np.sort(p_user)
    print(u_pop_sorted[-10:])



    fig = plt.figure(constrained_layout=True,figsize=(12,6))


    def plot_embed(ax1,ax2,idx):
        u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j]
        x1 = np.cos(u)*np.sin(v)
        y1 = np.sin(u)*np.sin(v)
        z1 = np.cos(v)
        ax1.plot_wireframe(x1, y1, z1, color="0.5",linewidth=0.1)
        user_idx=pop_users.index(idx)
        m_user=users[user_idx]
        target=np.array([1,-1,1])
        r_theta=np.arccos(np.dot(m_user,target)/(np.linalg.norm(m_user)*np.linalg.norm(target)))
        axis=np.cross(m_user,target)
        R=get_rotation_matrix(axis,r_theta)
        grp_theta=[]
        grp_r=[]
        sizes=[10,10,10]

        cmap_b = 'b'
        cmap_r = 'r'
        cmaps=[cmap_b,cmap_r]

        norm = plt.Normalize(vmin=-3, vmax=3)

        all_sampled=set([])
        all_pos=set([])
        for i,grp in enumerate(pop_group):
            sampled_group=set(np.random.choice(np.array(grp),50,replace=False).tolist())
            if user_idx in eval_test_id_split[i].keys():
                for item in eval_test_id_split[i][user_idx]:
                    sampled_group.add(item)
                    all_pos.add(item)
            for item in eval_train_split[i][user_idx]:
                sampled_group.add(item)
                all_pos.add(item)
            if user_idx in eval_test_ood_split[i].keys():
                for item in eval_test_ood_split[i][user_idx]:
                    sampled_group.add(item)
                    all_pos.add(item)
            
            all_sampled=all_sampled.union(sampled_group)
            

        all_neg=all_sampled.difference(all_pos)
        #print(all_neg)
        all_pos=np.array(list(all_pos),dtype=int)
        all_neg=np.array(list(all_neg),dtype=int)
        nor = plt.Normalize(vmin=-3, vmax=3)
        r=np.linalg.norm(target)

        lab=["neg","pos"]
        for i,idx in enumerate([all_neg,all_pos]):
            g_item=items[idx]
            g_item=np.matmul(g_item,R.T)
            norm=np.linalg.norm(g_item,axis=1)
            x=g_item[:,0]/norm#*r
            y=g_item[:,1]/norm#*r
            z=g_item[:,2]/norm#*r

            for j in range(len(idx)):
                ax1.plot([0,g_item[j][0]/norm[j]],[0,g_item[j][1]/norm[j]],[0,g_item[j][2]/norm[j]],color = cmaps[i],alpha=0.1)

            ax1.scatter(x, y, z, c = cmaps[i], marker =".",s=10,label=lab[i])

        
        #print("V^{T}",V_transpose)

        ax1.scatter(target[0]/r, target[1]/r, target[2]/r, c = 'g', marker ="*",s=120,label="user")
        ax1.plot([0,target[0]/r],[0,target[1]/r],[0,target[2]/r],color = 'g',alpha=0.1)
        ax1.legend()

        

        all_items=set([i for i in range(n_items)])
        all_neg=all_items.difference(all_pos)
        all_neg=np.array(list(all_neg),dtype=int)

        grp=["(neg):","(pos):"]
        txt=""

        for i,idx in enumerate([all_neg,all_pos]):
            g_item=items[idx]
            g_item=np.matmul(g_item,R.T)
            norm=np.linalg.norm(g_item,axis=1)
            cos=np.arccos(np.matmul(target,g_item.T)/norm/r)
            me=float(np.mean(cos))
            me=round(me,3)
            if i==1:
                txt="mean angle"+grp[i]+str(me)+"\n"+txt
            else:
                txt="mean angle"+grp[i]+str(me)+txt
            
            ax2.hist(cos,50,range=[0,np.pi],color=cmaps[i],weights=np.zeros_like(cos) + 1. / cos.size,edgecolor='black',alpha=0.6)
            mi_x,ma_x=ax2.get_xlim()
            mi_y,ma_y=ax2.get_ylim()
        ax2.text(mi_x+(ma_x-mi_x)*0.45, mi_y+(ma_y-mi_y)*0.75,txt , style ='italic') 
        red_patch = mpatches.Patch(color='red', alpha=0.6, label='pos')
        blue_patch = mpatches.Patch(color='blue', alpha=0.6,label='neg')
        ax2.legend(handles=[red_patch,blue_patch])
        
    
    pops=[205,30,10]


    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax2 = fig.add_subplot(gs[2,0:2])
    ax1.set_xticks([-1,-0.5,0,0.5,1])
    ax1.set_yticks([-1,-0.5,0,0.5,1])
    ax1.set_zticks([-1,-0.5,0,0.5,1])
    ax1.grid(False)

    plot_embed(ax1,ax2,pops[0])
    #ax1.set_title("High Pop User(p=205)")
    #ax2.set_title("Angular Distribution(High Pop)")

    plt.savefig(name+"high_pop_"+str(pops[0])+".png",bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax3 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax4 = fig.add_subplot(gs[2,0:2])
    ax3.set_xticks([-1,-0.5,0,0.5,1])
    ax3.set_yticks([-1,-0.5,0,0.5,1])
    ax3.set_zticks([-1,-0.5,0,0.5,1])
    ax3.grid(False)
    plot_embed(ax3,ax4,pops[1])
    #ax3.set_title("Mid Pop User(p=30)")
    #ax4.set_title("Angular Distribution(Mid Pop)")

    plt.savefig(name+"mid_pop_"+str(pops[1])+".png",bbox_inches='tight')
    plt.close()



    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax5 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax6 = fig.add_subplot(gs[2,0:2])
    ax5.set_xticks([-1,-0.5,0,0.5,1])
    ax5.set_yticks([-1,-0.5,0,0.5,1])
    ax5.set_zticks([-1,-0.5,0,0.5,1])
    ax5.grid(False)
    plot_embed(ax5,ax6,pops[2])
    #ax5.set_title("Low Pop User(p=10)")
    #ax6.set_title("Angular Distribution(Low Pop)")

    plt.savefig(name+"low_pop_"+str(pops[2])+".png",bbox_inches='tight')
    plt.close()

def seed_torch(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

if __name__ == '__main__':


    start = time.time()

    args = parse_args()
    seed_torch(args.seed)
    print(args)
    data = Data(args)
    data.load_data()
    device="cuda:"+str(args.cuda)
    device = torch.device(args.cuda)
    saveID = args.saveID
    saveID += str(args.dsc)
    #@ 保存的目录
    # need change 2
    if args.modeltype == "INFONCE":
        saveID += "_INFONCE_" + "n_layers=" + str(args.n_layers) + "tau=" + str(args.tau) + "batch_size=" + str(args.batch_size) + "neg_sample=" + str(args.neg_sample) + "lr=" + str(args.lr)
    if args.modeltype == "INFONCE_linear":
        saveID += "_INFONCE_linear_" + "n_layers=" + str(args.n_layers) + "tau=" + str(args.tau) + "batch_size=" + str(args.batch_size) + "neg_sample=" + str(args.neg_sample) + "lr=" + str(args.lr)

    if args.modeltype == "BC_LOSS":
        saveID += "n_layers=" + str(args.n_layers) + "tau1=" + str(args.tau1) + "tau2=" + str(args.tau2) + "w=" + str(args.w_lambda)
    
    if args.modeltype == "NBC_Loss":
        saveID += "n_layers=" + str(args.n_layers) + "tau1=" + str(args.tau1) + "tau2=" + str(args.tau2) + "w=" + str(args.w_lambda) + "k_score=" + str(args.k_score)

    if args.modeltype == "SimpleX":
        saveID += "_SimpleX_" + "n_layers=" + str(args.n_layers) + "tau=" + str(args.tau) + "batch_size=" + str(args.batch_size) + "neg_sample=" + str(args.neg_sample) + "lr=" + str(args.lr) + "w=" + str(args.w_neg) + "m=" + str(args.neg_margin)

    if args.modeltype == "XSimGCL":
        saveID += "_XSimGCL_" + "n_layers=" + str(args.n_layers) + "batch_size=" + str(args.batch_size) + "lr=" + str(args.lr) + "temp=" + str(args.temp_cl) + "layer_cl=" + str(args.layer_cl) + "lambda_cl=" + str(args.lambda_cl)+ "eps_XSimGCL=" + str(args.eps_XSimGCL)

    if args.modeltype == "SGL":
        saveID += "_SGL_" + "n_layers=" + str(args.n_layers) + "batch_size=" + str(args.batch_size) + "lr=" + str(args.lr) + "temp=" + str(args.temp_cl) + "lambda_cl=" + str(args.lambda_cl)+ "droprate=" + str(args.droprate)

    if args.modeltype == "NCL":
        saveID += "_NCL_" + "n_layers=" + str(args.n_layers) + "batch_size=" + str(args.batch_size) + "lr=" + str(args.lr) + "temp=" + str(args.temp_cl) + "lambda_cl=" + str(args.lambda_cl)+ "proto_reg=" + str(args.proto_reg) + "n_clusters=" + str(args.num_clusters) + "alpha=" + str(args.ncl_alpha)


    if args.n_layers > 0 and args.modeltype != "LGN":
        base_path = './weights/{}/{}-LGN/{}'.format(args.dataset, args.modeltype, saveID)
    else:
        base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    if args.modeltype == 'LGN':
        saveID += "n_layers=" + str(args.n_layers) + "lr=" + str(args.lr) + "batch_size=" + str(args.batch_size)
        base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    checkpoint_buffer=[]
    #@ 训练margin的epoch
    freeze_epoch=args.freeze_epoch if (args.modeltype=="BC_LOSS") else 0
    ensureDir(base_path)

    perf_str = str(args)
    with open(base_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            f.write(perf_str+"\n")

    #@ 计算popularity
    p_item = np.array([len(data.train_item_list[u]) if u in data.train_item_list else 0 for u in range(data.n_items)])
    p_user = np.array([len(data.train_user_list[u]) if u in data.train_user_list else 0 for u in range(data.n_users)])
    m_user=np.argmax(p_user)
    
    pop_sorted=np.sort(p_item)
    n_groups=3
    grp_view=[]
    for grp in range(n_groups):
        split=int((data.n_items-1)*(grp+1)/n_groups)
        grp_view.append(pop_sorted[split])
    print("group_view:",grp_view)
    idx=np.searchsorted(grp_view,p_item)

    if(args.dataset != "tencent_synthetic" and args.dataset != "kuairec_ood"):
        eval_test_ood_split=split_grp_view(data.test_ood_user_list,idx)
        eval_test_id_split=split_grp_view(data.test_id_user_list,idx)

    grp_view=[0]+grp_view

    pop_dict={}
    for user,items in data.train_user_list.items():
        for item in items:
            if item not in pop_dict:
                pop_dict[item]=0
            pop_dict[item]+=1
    # pop_dict的key是item，value是item的popularity
    
    sort_pop=sorted(pop_dict.items(), key=lambda item: item[1],reverse=True)
    # sort_pop是一个list，list中的元素是tuple，tuple的第一个元素是item，第二个元素是item的popularity
    pop_mask=[item[0] for item in sort_pop[:20]]
    print(pop_mask)

    if "kuairec" in args.dataset:
        with open("data/" + args.dataset + '/not_candidate.txt', 'r') as f:
            not_candidate = f.readlines()
            not_candidate = [int(item.strip()) for item in not_candidate]
            not_candidate_dict = {u:not_candidate for u in data.users}


    if not args.pop_test:
        
        if(args.dataset == "tencent_synthetic"):
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20])
            eval_test_ood_1 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_1,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_2,data.test_ood_user_list_3]))
            eval_test_ood_2 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_2,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_1,data.test_ood_user_list_3]))
            eval_test_ood_3 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_3,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_1,data.test_ood_user_list_2]))
        elif(args.dataset == "kuairec_ood"):
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,not_candidate_dict]))
            eval_test_ood_1 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_1,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,not_candidate_dict]))
            eval_test_ood_2 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_2,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,not_candidate_dict, data.test_ood_user_list_1]))
            eval_test_ood_3 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_3,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,not_candidate_dict, data.test_ood_user_list_1, data.test_ood_user_list_2]))      
        else:
            if "kuairec" in args.dataset:
                eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,not_candidate_dict]))
                eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list,not_candidate_dict]))
                eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list,not_candidate_dict]))
            else: 
                eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20])  
                eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]))
                eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]))
        
    else:
        if(args.dataset == "tencent_synthetic"):
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],pop_mask=pop_mask)
            eval_test_ood_1 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_1,top_k=[20],\
                            dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_2,data.test_ood_user_list_3]),pop_mask=pop_mask)
            eval_test_ood_2 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_2,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_1,data.test_ood_user_list_3]),pop_mask=pop_mask)
            eval_test_ood_3 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_3,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list_1,data.test_ood_user_list_2]),pop_mask=pop_mask)
        elif(args.dataset == "kuairec_ood"):
            eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],pop_mask=pop_mask,dump_dict=merge_user_list([data.train_user_list,not_candidate_dict]))
            eval_test_ood_1 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_1,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,not_candidate_dict]))
            eval_test_ood_2 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_2,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,not_candidate_dict,data.test_ood_user_list_1]))
            eval_test_ood_3 = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list_3,top_k=[20],\
                                dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,not_candidate_dict,data.test_ood_user_list_1,data.test_ood_user_list_2]))
        else:
            if "kuairec" in args.dataset:
                eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,not_candidate_dict]))
                eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list,not_candidate_dict]))
                eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list,not_candidate_dict]))
            else:
                eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],pop_mask=pop_mask)
                eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]),pop_mask=pop_mask)
                eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]),pop_mask=pop_mask)

    if(args.dataset == "tencent_synthetic" or args.dataset == "kuairec_ood"):
        evaluators=[eval_valid, eval_test_ood_1, eval_test_ood_2, eval_test_ood_3]
        eval_names=["valid","test_ood_1", "test_ood_2", "test_ood_3"]
    else:
        evaluators=[ eval_valid,eval_test_id, eval_test_ood]
        eval_names=["valid","test_id", "test_ood" ]

    # need change 3
    if args.modeltype == 'LGN':
        model = LGN(args, data)
    if args.modeltype == 'INFONCE':
        model = INFONCE(args, data)
    if args.modeltype == 'INFONCE_linear':
        model = INFONCE_linear(args, data)
    if args.modeltype == 'IPS':
        model = IPS(args, data)
    if args.modeltype == 'CausE':
        model = CausE(args, data)
    if args.modeltype == 'BC_LOSS':
        model = BC_LOSS(args, data)
    if args.modeltype == 'MACR':
        model = MACR(args, data)
    if args.modeltype == 'SAMREG':
        model = SAMREG(args, data)
    if args.modeltype == "SimpleX":
        model = SimpleX(args,data)
    if args.modeltype == "SGL":
        model = SGL(args,data)
    if args.modeltype == "NCL":
        model = NCL(args,data)
    if args.modeltype == "XSimGCL":
        model = XSimGCL(args,data)
    if args.modeltype == "PID":
        model = PID(args,data)
    if args.modeltype == "NBC_LOSS":
        model = NBC_LOSS(args, data)
    
#    b=args.sample_beta

    model.cuda(device)

    #@ 读取模型
    model, start_epoch = restore_checkpoint(model, base_path, device)

    if args.test_only:

        for i,evaluator in enumerate(evaluators):
            is_best, temp_flag = evaluation(args, data, model, start_epoch, base_path, evaluator,eval_names[i])

        exit()
                

    flag = False
    
    # for param in model.named_parameters():
        # print(param)
    if args.modeltype == "PID":
        optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=5e-6)
    else:
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=model.lr)

    #item_pop_idx = torch.tensor(data.item_pop_idx).cuda(device)

    
    for epoch in range(start_epoch, args.epoch):
        print(f"current epoch: {epoch}/{args.epoch}")

        # If the early stopping has been reached, restore to the best performance model
        if flag:
            break
        
        # need change 4
        # All models
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        # CausE
        running_cf_loss = 0
        # BC_LOSS
        running_loss1, running_loss2 = 0, 0
        # XSimGCL
        running_cl_loss = 0
        # NCL
        running_proto_loss = 0
        # PID
        running_pid_loss = 0

        if args.modeltype == "SGL":
            print("get enhanced graph")
            dropped_adj1 = model.get_enhanced_adj(data.ui_mat, args.droprate)
            dropped_adj2 = model.get_enhanced_adj(data.ui_mat, args.droprate)
        
        if args.modeltype == "NCL" and epoch >= args.ncl_start_epoch:
            model.e_step()

        t1=time.time()

        pbar = tqdm(enumerate(data.train_loader), mininterval=2, total = len(data.train_loader))

        for batch_i, batch in pbar: 
            # print(batch_i, batch)           

            batch = [x.cuda(device) for x in batch]

            users = batch[0]
            pos_items = batch[1]

            if args.modeltype != 'CausE':
                users_pop = batch[2]
                pos_items_pop = batch[3]
                pos_weights = batch[4]
                if args.infonce == 0 or args.neg_sample != -1:
                    neg_items = batch[5]
                    neg_items_pop = batch[6]

            model.train()

            # need change 5

            if args.modeltype == 'INFONCE':

                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss
            
            elif args.modeltype == 'INFONCE_linear':

                mf_loss, reg_loss = model(users, pos_items, neg_items, neg_items_pop)
                loss = mf_loss + reg_loss

            elif args.modeltype == 'BC_LOSS':
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm  = model(users, pos_items, neg_items, \
                                                                                users_pop, pos_items_pop, neg_items_pop)
                
                if epoch < args.freeze_epoch:
                    loss =  loss2 + reg_loss_freeze #@ loss2是poplarity bias loss
                else:
                    model.freeze_pop()
                    loss = loss1 + loss2 + reg_loss
            
            elif args.modeltype == 'NBC_LOSS':
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm  = model(users, pos_items, neg_items, \
                                                                                users_pop, pos_items_pop, neg_items_pop)
                
                if epoch < args.freeze_epoch:
                    loss = loss1 + loss2 + reg_loss_freeze
                else:
                    model.freeze_pop()
                    loss = loss1 + loss2 + reg_loss

            elif args.modeltype == 'IPS' or args.modeltype =='SAMREG':

                mf_loss, reg_loss = model(users, pos_items, neg_items, pos_weights)
                loss = mf_loss + reg_loss

            elif args.modeltype == 'CausE':
                neg_items = batch[2]
                all_reg = torch.squeeze(batch[3].T.reshape([1, -1]))
                all_ctrl = torch.squeeze(batch[4].T.reshape([1, -1]))
                mf_loss, reg_loss, cf_loss = model(users, pos_items, neg_items, all_reg, all_ctrl)
                loss = mf_loss + reg_loss + cf_loss 
            
            elif args.modeltype == "SimpleX":
                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss

            elif args.modeltype == "XSimGCL":
                mf_loss, cl_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + cl_loss + reg_loss

            elif args.modeltype == "SGL":
                mf_loss, cl_loss, reg_loss = model(users, pos_items, neg_items, dropped_adj1, dropped_adj2)
                loss = mf_loss + cl_loss + reg_loss

            elif args.modeltype == "NCL":
                mf_loss, cl_loss, proto_loss, reg_loss = model(users, pos_items, neg_items, epoch)
                if(epoch >= args.ncl_start_epoch):
                    loss = mf_loss + cl_loss + proto_loss + reg_loss
                else:
                    loss = mf_loss + cl_loss + reg_loss

            elif args.modeltype == "PID":
                mf_loss, pid_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + pid_loss + reg_loss
                # loss = mf_loss + reg_loss 

            else:
                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()

            # need change 6
            if args.modeltype != 'BC_LOSS' and args.modeltype != 'NBC_LOSS':
                running_mf_loss += mf_loss.detach().item()
            
            if args.modeltype == 'CausE':
                running_cf_loss += cf_loss.detach().item()

            if args.modeltype == 'BC_LOSS' or args.modeltype == 'NBC_LOSS':
                running_loss1 += loss1.detach().item()
                running_loss2 += loss2.detach().item()

            if args.modeltype == 'XSimGCL' or args.modeltype == 'SGL':
                running_cl_loss += cl_loss.detach().item()
            
            if args.modeltype == 'NCL':
                running_cl_loss += cl_loss.detach().item()
                running_proto_loss += proto_loss.detach().item()
            
            if args.modeltype == 'PID':
                running_pid_loss += pid_loss.detach().item()

            num_batches += 1

        t2=time.time()

        # need change 7
        # Training data for one epoch
        if args.modeltype == "CausE":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_reg_loss / num_batches, running_cf_loss / num_batches)
        
        elif args.modeltype=="BC_LOSS" or args.modeltype=="NBC_LOSS":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_loss1 / num_batches, running_loss2 / num_batches, running_reg_loss / num_batches)
                #@ 1是main branch 2是popularity branch 3是reg branch

        elif args.modeltype == "XSimGCL" or args.modeltype == "SGL":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_cl_loss / num_batches, running_reg_loss / num_batches)
        
        elif args.modeltype == "NCL":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_cl_loss / num_batches, running_proto_loss / num_batches, running_reg_loss / num_batches)

        elif args.modeltype == "PID":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_pid_loss / num_batches, running_reg_loss / num_batches)

        else:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_reg_loss / num_batches)

        #@ 表现写入txt文件
        with open(base_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            f.write(perf_str+"\n")

        # Evaluate the trained model
        if (epoch + 1) % args.verbose == 0 and epoch >= freeze_epoch:
            model.eval() 

            for i,evaluator in enumerate(evaluators):
                is_best, temp_flag = evaluation(args, data, model, epoch, base_path, evaluator,eval_names[i])
                
                if is_best:
                    checkpoint_buffer=save_checkpoint(model, epoch, base_path, checkpoint_buffer, args.max2keep)
                
                #@ early stop?
                if temp_flag:
                    flag = True
            
            all_users, all_items = model.compute()
            users_emb = all_users[users]
            pos_emb = all_items[pos_items]
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            aloss = align_loss(users_emb, pos_emb)
            uloss = uniform_loss(users_emb)
            uloss += uniform_loss(pos_emb)

            with open(base_path + '/2loss.txt','a') as f:
                f.write(str(aloss.item())+"\t"+str(uloss.item())+"\n")

            model.train()
        
    # Get result
    model = restore_best_checkpoint(data.best_valid_epoch, model, base_path, device)
    print_str = "The best epoch is % d" % data.best_valid_epoch
    with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")

    for i,evaluator in enumerate(evaluators[:]):
        evaluation(args, data, model, epoch, base_path, evaluator, eval_names[i])
    with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")

    read_log(base_path +'stats_{}.txt'.format(args.saveID))

