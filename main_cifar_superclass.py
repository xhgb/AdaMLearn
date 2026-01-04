import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from deepcore.nets import LeNet
import deepcore.methods as methods
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import argparse,time
from copy import deepcopy
from utils import str_to_bool


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor  

def train(args, model, device, x,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)        
        loss.backward()
        optimizer.step()

def train_projected(args,model,device,x,y,optimizer,criterion,feature_mat,task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)         
        loss.backward()
        # Gradient Projections 
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if k<4 and len(params.size())!=1:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                        feature_mat[kk]).view(params.size())
                kk +=1
            elif (k<4 and len(params.size())==1) and task_id !=0 :
                params.grad.data.fill_(0)

        optimizer.step()

def tst(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def get_activation(name, input_data):
    def hook(model, input, output):
        input_data[name] = input[0].detach()  # input type is tulple, only has one element, which is the tensor
    return hook


def get_representation_matrix (net, device, x, y, dst_train, task_id):
    act_dict = OrderedDict()
    map = [32, 16]
    ksize = [5, 5]
    in_channel = [3, 40]
    # Collect activations by forward pass
    if args.core_set_selection:
        selection_args = dict(selection_method=args.uncertainty,
                              balance=args.balance,
                              greedy=args.submodular_greedy,
                              function=args.submodular)
        method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
        subset = method.directly_select(net, task_id)
        example_num = len(subset['indices'])
        example_data = x[subset['indices']]
    else:
        example_num = args.example_num
        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).to(device)
        b = r[0:example_num]  # Take random training samples
        example_data = x[b]

    # hook the specific layers' input
    net.features[0].register_forward_hook(get_activation('conv_1', act_dict))
    net.features[5].register_forward_hook(get_activation('conv_2', act_dict))
    net.features_lin[0].register_forward_hook(get_activation('fc_1', act_dict))
    net.features_lin[3].register_forward_hook(get_activation('fc_2', act_dict))

    example_data = example_data.to(device)
    example_out  = net(example_data)
    batch_list = [example_num, example_num, example_num, example_num]
    pad = 2
    p1d = (2, 2, 2, 2)
    mat_list=[]
    act_key=list(act_dict.keys())
    # pdb.set_trace()
    for i in range(len(act_key)):
        bsz=batch_list[i]
        k=0
        if i<2:
            ksz = ksize[i]
            s=compute_conv_output_size(map[i],ksz,1,pad)
            mat = np.zeros((ksize[i]*ksize[i]*in_channel[i],s*s*bsz))
            act = F.pad(act_dict[act_key[i]], p1d, "constant", 0).detach().cpu().numpy()
         
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) #?
                        k +=1
            mat_list.append(mat)
        else:
            act = act_dict[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list    

def update_memory (model, mat_list, threshold, feature_list=[]):
    print ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_pro = np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
            act_hat = activation - act_pro
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating memory for layer: {}'.format(i+1))
                continue
            # update memory
            U2, S2, Vh2 = np.linalg.svd(act_pro, full_matrices=False)
            sval_total2 = (S2 ** 2).sum()
            sval_ratio2 = (S2 ** 2) / sval_total2
            rank = np.sum(np.cumsum(sval_ratio2) < args.active_forget)
            feature_forgetted = U2[:, 0:rank]
            Ui = np.hstack((feature_forgetted, U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list  


def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Choose any task order - ref {yoon et al. ICLR 2020}
    task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                  np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                  np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                  np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]

    ## Load CIFAR100_SUPERCLASS DATASET
    from dataloader import cifar100_superclass as data_loader
    data, taskcla = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5, validation=True)
    test_data,_   = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5)
    print (taskcla)

    acc_matrix=np.zeros((20,20))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    for k,ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([0.98] * 5) + task_id*np.array([0.001] * 5)
     
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x'].to(device)
        ytrain=data[k]['train']['y'].to(device)
        xvalid=data[k]['valid']['x'].to(device)
        yvalid=data[k]['valid']['y'].to(device)
        xtest =test_data[k]['test']['x'].to(device)
        ytest =test_data[k]['test']['y'].to(device)
        dst_train = xtrain
        dst_train.classes = range(5)
        dst_train.targets = data[k]['train']['y'].cpu()

        lr = args.lr 
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)
        
        if task_id==0:
            model = LeNet(channel=3, num_classes=5, backpack=args.backpack).to(device)
            print ('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                print (k_t,m,param.shape)
            print ('-'*40)
            # Initialize model 
            model.apply(init_weights)
            best_model=get_model(model)
            feature_list =[]
            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = tst(args, model, device, xtrain, ytrain, criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
                # Validate
                valid_loss,valid_acc = tst(args, model, device, xvalid, yvalid, criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<args.lr_min:
                            print()
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()
            set_model_(model,best_model)
            # Test
            print ('-'*40)
            test_loss, test_acc = tst(args, model, device, xtest, ytest, criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list = get_representation_matrix (model, device, xtrain, ytrain, dst_train, k)
            feature_list = update_memory (model, mat_list, threshold, feature_list)

        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(feature_list)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)
            print ('-'*40)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,feature_mat,k)
                clock1=time.time()
                tr_loss, tr_acc = tst(args, model, device, xtrain, ytrain, criterion, k)
                print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)),end='')
                # Validate
                valid_loss,valid_acc = tst(args, model, device, xvalid, yvalid, criterion, k)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<args.lr_min:
                            print()
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer, epoch, args)
                print()
            set_model_(model,best_model)
            # Test 
            test_loss, test_acc = tst(args, model, device, xtest, ytest, criterion, k)
            print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  
            # Memory Update 
            mat_list = get_representation_matrix (model, device, xtrain, ytrain, dst_train, k)
            feature_list = update_memory (model, mat_list, threshold, feature_list)
        
        # save accuracy 
        jj = 0 
        for ii in task_order[args.t_order][0:task_id+1]:
            xtest =test_data[ii]['test']['x'].to(device)
            ytest =test_data[ii]['test']['y'].to(device)
            _, acc_matrix[task_id,jj] = tst(args, model, device, xtest, ytest, criterion, ii)
            jj +=1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()
        # update task id 
        task_id +=1
    print('-'*50)
    # Simulation Results 
    print ('Task Order : {}'.format(task_order[args.t_order]))
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index = [i for i in ["1","2","3","4","5","6","7",\
                                        "8","9","10","11","12","13","14","15","16","17","18","19","20"]],
                      columns = [i for i in ["1","2","3","4","5","6","7",\
                                        "8","9","10","11","12","13","14","15","16","17","18","19","20"]])
    sn.set(font_scale=1.4) 
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential CIFAR-superclass')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--t_order', type=int, default=0, metavar='TOD',
                        help='random seed (default: 0)')
    parser.add_argument('--active_forget', default=0.9992, type=float,
                        help='proportion of active_forget, 0.9992')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # core-set selection
    parser.add_argument('--core_set_selection', type=bool, default=True, help="use core-set selection or not")
    parser.add_argument('--backpack', type=str_to_bool, default=True, help='whether to utilize backpack or not')
    parser.add_argument('--fraction', default=0.01, type=float,
                        help='fraction of data to be selected (default: 0.01)')
    parser.add_argument('--example_num', default=25, type=int,
                        help='random selected samples (default: 25)')
    parser.add_argument('--selection', type=str, default="Submodular", help="selection method")
    parser.add_argument("--selection_batch", "-sb", default=1, type=int,
                        help="batch size for selection, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument('--selection_optimizer', "-so", default="SGD",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool,
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="LazyGreedy",
                        help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")
    parser.add_argument('--model_name', type=str, default='LeNet', help='model')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--workers', type=int, default='0', help='num_workers')
    parser.add_argument('--kernel', type=str, default='worst', help="path to latest checkpoint (default: do not load)")
    parser.add_argument('--exact_analyses', type=str_to_bool, default=False, help='whether to utilize backpack or not')
    parser.add_argument('--K', type=int, default=100,
                        help='how many sub-dimensions to choose from the whole parameter dimension')
    parser.add_argument('--eps', type=float, default=0.01, help="path to latest checkpoint (default: do not load)")
    parser.add_argument('--print_freq', '-p', default=128, type=int, help='print frequency (default: 20)')


    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)



