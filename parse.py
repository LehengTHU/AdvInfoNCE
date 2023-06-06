import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', nargs='?', default=-1,
                        help='we only want test value.')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--test_only', nargs='?', default=False,
                        help='we only want test value.')
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='yahoo.new',
                        help='Choose a dataset')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularization.')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Number of epoch.')
    parser.add_argument('--Ks', nargs='?', default= [20],
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log\'s interval epoch while training')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--saveID', type=str, default="",
                        help='Specify model save path.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping point.')
    parser.add_argument('--checkpoint', type=str, default='./',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'BC_LOSS',
                        help='Specify model save path.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--IPStype', type=str, default='cn',
                        help='Specify the mode of weighting')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--codetype', type=str, default='train',
                        help='Calculate overlap with Item pop')
    parser.add_argument('--max2keep', type=int, default=10,
                        help='max checkpoints to keep')
    parser.add_argument('--infonce', type=int, default=1,
                        help='whether to use infonce loss or not')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in data loader')
    parser.add_argument('--dsc', type=str, default='',
                        help='describe the code version')
    parser.add_argument("--train_norm", action="store_true",
                        help="train_norm")
    parser.add_argument("--pred_norm", action="store_true",
                        help="pred_norm")
    
                        
                        
    # INFONCE
    parser.add_argument('--tau', type=float, default=0.1,
                        help='temperature parameter')
    parser.add_argument('--eta', type=float, default=1,
                        help='eta')

    # MACR
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help='alpha')
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='beta')
    parser.add_argument('--c', type=float, default=30.0,
                        help='Constant c.')
    #CausE
    parser.add_argument('--cf_pen', type=float, default=0.05,
                        help='Imbalance loss.')
    
    #SAM-REG

    parser.add_argument('--rweight', type=float, default=0.05)
    parser.add_argument('--sam',type=bool,default=True)
    parser.add_argument('--pop_test',type=bool,default=False)

    #SimpleX

    parser.add_argument('--w_neg', type=float, default=1)
    parser.add_argument('--neg_margin',type=float, default=0.4)
    
    #BC_LOSS
    parser.add_argument('--tau1', type=float, default=0.07,
                        help='temperature parameter for L1')
    parser.add_argument('--tau2', type=float, default=0.1,
                        help='temperature parameter for L2')
    parser.add_argument('--w_lambda', type=float, default=0.5,
                        help='weight for combining l1 and l2.')
    parser.add_argument('--freeze_epoch',type=int,default=5)

    parser.add_argument('--neg_sample',type=int,default=128)

    #AdvInfoNCE
    parser.add_argument('--adv_interval',type=int,default=5,
                        help='the interval of adversarial training')
    parser.add_argument('--adv_epochs',type=int,default=1,
                        help='the epoch of adversarial training')
    parser.add_argument('--warm_up_epochs', type=int, default=0,
                        help='warm up epochs, in this stage, adv training is not used')
    parser.add_argument('--adv_lr', type=float, default=5e-5,
                        help='Learning rate for adversarial training.')
    parser.add_argument('--k_neg', type=float, default=64,
                        help='k_neg for negative sampling')
    parser.add_argument('--eta_epochs', type=int, default=7,
                        help='epochs for eta, control the disturbance of adv training')
    parser.add_argument('--model_version', type=str, default='embed',
                        help='model type, mlp or embed')
    parser.add_argument('--w_embed_size',type=int,default=64,
                        help='dimension of weight embedding')


    # Contrastive Learning
    parser.add_argument('--lambda_cl', type=float, default=0.2,
                        help='Rate of contrastive loss')
    parser.add_argument('--temp_cl', type=float, default=0.15,
                        help='Temperature of contrastive loss')
    
    # SGL
    parser.add_argument('--droprate', type=float, default=0.1,
                        help='drop out rate for SGL')

    # NCL
    parser.add_argument('--proto_reg', type=float, default=1e-7,
                        help='regularization for prototype')
    parser.add_argument('--ncl_alpha', type=float, default=1,
                        help='alpha for ncl')
    parser.add_argument('--num_clusters', type=int, default=2000,
                        help='number of clusters')
    parser.add_argument('--ncl_start_epoch', type=int, default=20,
                        help='start epoch for ncl')
    
    

    # XSimGCL
    parser.add_argument('--eps_XSimGCL', type=float, default=0.2,
                        help='Noise rate')
    parser.add_argument('--layer_cl',type=int,default=1,
                        help='Which layer to pick the contrastive view')

    # PID
    parser.add_argument('--alpha_pid', type=float, default=0.8,
                        help='alpha for pid')
    
    # NBC Loss
    parser.add_argument('--k_score', type=float, default=8,
                        help='Temperature of contrastive loss')

    # p_distribution
    parser.add_argument('--p_path', type=str, default='',
                        help='path of p distribution')

    # Adap-tau
    parser.add_argument('--cnt_lr',type=int,default=100,
                        help='warm up for adap tau')
    parser.add_argument('--adap_tau_beta', type=float, default=1.0,
                        help='beta')
    
    return parser.parse_args()


