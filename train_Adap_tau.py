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
from model import Adap_tau
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter
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
        inp_epoch = 0
        # inp_epoch = epoch
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


def ensureDir(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def checktensor(tensor):
    t=tensor.detach().cpu().numpy()
    if np.max(np.isnan(t)):        
        idx=np.argmax(np.isnan(t))
        return idx
    else:
        return -1

def split_grp_view(data,grp_idx):
    n=len(grp_view)
    split_data=[{} for _ in range(n)]

    for key,item in data.items():
        for it in item:
            if key not in split_data[grp_idx[it]].keys():
                split_data[grp_idx[it]][key]=[]
            split_data[grp_idx[it]][key].append(it)
    return split_data

def seed_torch(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
    saveID += "_Adap_tau_" + str(args.dataset) + '_n_layers_' + str(args.n_layers) + '_lr_' + str(args.lr)  + '_batch_' + str(args.batch_size) + '_warm_up_' + str(args.cnt_lr) + '_tau_' + str(args.tau)  + '_patience_' + str(args.patience)

    if args.n_layers > 0:
        base_path = './weights/{}/{}-LGN/{}'.format(args.dataset, args.modeltype, saveID)
    else:
        base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    checkpoint_buffer=[]
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


    model = Adap_tau(args,data)
    model.cuda(device)
    #@ 读取模型
    model, start_epoch = restore_checkpoint(model, base_path, device)

    if args.test_only:

        for i,evaluator in enumerate(evaluators):
            is_best, temp_flag = evaluation(args, data, model, start_epoch, base_path, evaluator,eval_names[i])

        exit()
                

    flag = False

    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=model.lr)

    train_cf = torch.LongTensor(np.array([np.array(data.trainUser), np.array(data.trainItem)], np.int32).T)
    
    loss_per_user = None
    loss_per_ins = None
    # prepare for tau_0
    pos = train_cf.to(device)
    nu = scatter(torch.ones(len(train_cf), device=device), pos[:, 0], dim=0, reduce='sum') #每个用户交互的物品数量
    nu_thresh = torch.quantile(nu, 0.2)
    judgeid_torch = (nu > nu_thresh)
    [useid_torch, ] = torch.where(judgeid_torch > 0)
    [yid_torch ,] = torch.where(judgeid_torch[pos[:,0]]>0)


    for epoch in range(start_epoch, args.epoch):
        print(f"current epoch: {epoch}/{args.epoch}")

        # If the early stopping has been reached, restore to the best performance model
        if flag:
            break

        losses_train = []
        tau_maxs = []
        tau_mins = []
        losses_emb = []
        hits = 0
        
        if epoch >= args.cnt_lr:
            user_emb_cos, item_emb_cos = model.compute()

            user_emb_cos = F.normalize(user_emb_cos, dim=-1).detach()
            item_emb_cos = F.normalize(item_emb_cos, dim=-1).detach()

            pos_scores = (user_emb_cos[pos[:, 0]] * item_emb_cos[pos[:, 1]]).sum(dim=-1)
            pos_u_torch = pos_scores[yid_torch].mean() #miu+
            # pos_var_torch = pos_scores[yid_torch].var()
            ev_mean_torch = item_emb_cos.mean(dim=0, keepdim=True) # item 做平均之后的一个平均的item值
            allu_torch = (user_emb_cos[useid_torch] @ ev_mean_torch.t()).view(-1)
            au_torch = allu_torch.mean() #miu
            can_torch = np.log(len(useid_torch) * data.n_items)
            a_torch = 1e-10
            c_torch = 2 * (np.log(0.5)+can_torch-np.log(len(yid_torch)))
            b_torch = - (pos_u_torch - au_torch)
            # w_torch = c_torch / (-2 * b_torch)
            w_0 = c_torch / (-2 * b_torch)
            # logger.info("current w_0 is {}".format(w_0.item()))
        else:
            # can = np.log(len(useid_torch) * data.n_items);
            # a = 1e-10;
            # c = 2 * (np.log(0.5) + can - np.log(len(yid_torch)))
            # b = - 0.7
            # w_0 = ( - b - np.sqrt(np.clip(b ** 2 - a*c , 0, 100000))) / a  # 固定一个tau_0
            w_0 = torch.tensor(1/args.tau)
            # logger.info("current w_0 is {}".format(w_0))
        
        print("current tau_0 is {}".format(1/w_0))


        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0

        t1=time.time()

        pbar = tqdm(enumerate(data.train_loader), mininterval=2, total = len(data.train_loader))

        for batch_i, batch in pbar: 
            # print(batch_i, batch)           

            batch = [x.cuda(device) for x in batch]

            users = batch[0]
            pos_items = batch[1]
            users_pop = batch[2]
            pos_items_pop = batch[3]
            pos_weights = batch[4]
            if args.infonce == 0 or args.neg_sample != -1:
                neg_items = batch[5]
                neg_items_pop = batch[6]
            
            if(batch_i == 0):
                train_cf_ = users
            else:
                train_cf_ = torch.cat((train_cf_, users))

            model.train()

            # loss_per_user = None
            mf_loss, mf_loss_, reg_loss, tau = model(users, pos_items, neg_items, loss_per_user, w_0=w_0, s=batch_i)
            loss = mf_loss - reg_loss
            # loss = mf_loss
            # loss = reg_loss
            tau_maxs.append(tau.max().item())
            tau_mins.append(tau.min().item())
            losses_train.append(mf_loss_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()

            running_mf_loss += mf_loss.detach().item()

            num_batches += 1
        
        losses_train = torch.cat(losses_train, dim=0)
        # loss_per_user = scatter(losses_train, train_cf_, dim=0, reduce='mean').detach() #算每个user的loss
        loss_per_user = scatter(losses_train, train_cf_, dim=0, reduce='mean')
        # loss_per_user = None

        t2=time.time()

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
            epoch, t2 - t1, running_loss / num_batches,
            running_mf_loss / num_batches, running_reg_loss / num_batches) + ' TAU_0:{:.4}, TAU_u:{:.4} {:.4}'.format(
                        1/w_0, 1/np.mean(tau_mins), 1/np.mean(tau_maxs)) + ' W_0:{:.4}, W_u:{:.4} {:.4}'.format(
                        w_0, np.mean(tau_mins), np.mean(tau_maxs))

        #@ 表现写入txt文件
        with open(base_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            f.write(perf_str+"\n")
        


        # Evaluate the trained model
        if (epoch + 1) % args.verbose == 0:
            model.eval() 

            for i,evaluator in enumerate(evaluators):
                is_best, temp_flag = evaluation(args, data, model, epoch, base_path, evaluator,eval_names[i])
                
                if is_best:
                    checkpoint_buffer=save_checkpoint(model, epoch, base_path, checkpoint_buffer, args.max2keep)
                
                #@ early stop?
                if temp_flag:
                    flag = True

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





