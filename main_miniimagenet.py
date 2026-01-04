import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from deepcore.nets.resnet import ResNet_84x84, BasicBlock
import deepcore.methods as methods
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import argparse,time
from copy import deepcopy
from utils import str_to_bool

class_order0 = np.array([26, 86,  2, 55, 75, 93, 16, 73, 54, 95, 53, 92, 78, 13,  7, 30, 22,
       24, 33,  8, 43, 62,  3, 71, 45, 48,  6, 99, 82, 76, 60, 80, 90, 68,
       51, 27, 18, 56, 63, 74,  1, 61, 42, 41,  4, 15, 17, 40, 38,  5, 91,
       59,  0, 34, 28, 50, 11, 35, 23, 52, 10, 31, 66, 57, 79, 85, 32, 84,
       14, 89, 19, 29, 49, 97, 98, 69, 20, 94, 72, 77, 25, 37, 81, 46, 39,
       65, 58, 12, 88, 70, 87, 36, 21, 83,  9, 96, 67, 64, 47, 44])
class_order1 = np.array([80, 84, 33, 81, 93, 17, 36, 82, 69, 65, 92, 39, 56, 52, 51, 32, 31,
       44, 78, 10,  2, 73, 97, 62, 19, 35, 94, 27, 46, 38, 67, 99, 54, 95,
       88, 40, 48, 59, 23, 34, 86, 53, 77, 15, 83, 41, 45, 91, 26, 98, 43,
       55, 24,  4, 58, 49, 21, 87,  3, 74, 30, 66, 70, 42, 47, 89,  8, 60,
        0, 90, 57, 22, 61, 63,  7, 96, 13, 68, 85, 14, 29, 28, 11, 18, 20,
       50, 25,  6, 71, 76,  1, 16, 64, 79,  5, 75,  9, 72, 12, 37])
class_order2 = np.array([83, 30, 56, 24, 16, 23,  2, 27, 28, 13, 99, 92, 76, 14,  0, 21,  3,
       29, 61, 79, 35, 11, 84, 44, 73,  5, 25, 77, 74, 62, 65,  1, 18, 48,
       36, 78,  6, 89, 91, 10, 12, 53, 87, 54, 95, 32, 19, 26, 60, 55,  9,
       96, 17, 59, 57, 41, 64, 45, 97,  8, 71, 94, 90, 98, 86, 80, 50, 52,
       66, 88, 70, 46, 68, 69, 81, 58, 33, 38, 51, 42,  4, 67, 39, 37, 20,
       31, 63, 47, 85, 93, 49, 34,  7, 75, 82, 43, 22, 72, 15, 40])
class_order3 = np.array([93, 67,  6, 64, 96, 83, 98, 42, 25, 15, 77,  9, 71, 97, 34, 75, 82,
       23, 59, 45, 73, 12,  8,  4, 79, 86, 17, 65, 47, 50, 30,  5, 13, 31,
       88, 11, 58, 85, 32, 40, 16, 27, 35, 36, 92, 90, 78, 76, 68, 46, 53,
       70, 80, 61, 18, 91, 57, 95, 54, 55, 28, 52, 84, 89, 49, 87, 37, 48,
       33, 43,  7, 62, 99, 29, 69, 51,  1, 60, 63,  2, 66, 22, 81, 26, 14,
       39, 44, 20, 38, 94, 10, 41, 74, 19, 21,  0, 72, 56,  3, 24])
class_order4 = np.array([20, 10, 96, 16, 63, 24, 53, 97, 41, 47, 43,  2, 95, 26, 13, 37, 14,
       29, 35, 54, 80,  4, 81, 76, 85, 60,  5, 70, 71, 19, 65, 62, 27, 75,
       61, 78, 18, 88,  7, 39,  6, 77, 11, 59, 22, 94, 23, 12, 92, 25, 83,
       48, 17, 68, 31, 34, 15, 51, 86, 82, 28, 64, 67, 33, 45, 42, 40, 32,
       91, 74, 49,  8, 30, 99, 66, 56, 84, 73, 79, 21, 89,  0,  3, 52, 38,
       44, 93, 36, 57, 90, 98, 58,  9, 50, 72, 87,  1, 69, 55, 46])


## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


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
            if len(params.size())==4:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                    feature_mat[kk]).view(params.size())
                kk+=1
            elif len(params.size())==1 and task_id !=0:
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


def get_representation_matrix_ResNet18 (net, device, x, y, dst_train, task_id):
    # Collect activations by forward pass
    net.eval()
    act_dict = OrderedDict()
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
    net.conv1.register_forward_hook(get_activation('conv_1', act_dict))
    net.layer1[0].conv1.register_forward_hook(get_activation('conv_2', act_dict))
    net.layer1[0].conv2.register_forward_hook(get_activation('conv_3', act_dict))
    net.layer1[1].conv1.register_forward_hook(get_activation('conv_4', act_dict))
    net.layer1[1].conv2.register_forward_hook(get_activation('conv_5', act_dict))

    net.layer2[0].conv1.register_forward_hook(get_activation('conv_6', act_dict))
    net.layer2[0].conv2.register_forward_hook(get_activation('conv_7', act_dict))
    net.layer2[1].conv1.register_forward_hook(get_activation('conv_8', act_dict))
    net.layer2[1].conv2.register_forward_hook(get_activation('conv_9', act_dict))

    net.layer3[0].conv1.register_forward_hook(get_activation('conv_10', act_dict))
    net.layer3[0].conv2.register_forward_hook(get_activation('conv_11', act_dict))
    net.layer3[1].conv1.register_forward_hook(get_activation('conv_12', act_dict))
    net.layer3[1].conv2.register_forward_hook(get_activation('conv_13', act_dict))

    net.layer4[0].conv1.register_forward_hook(get_activation('conv_14', act_dict))
    net.layer4[0].conv2.register_forward_hook(get_activation('conv_15', act_dict))
    net.layer4[1].conv1.register_forward_hook(get_activation('conv_16', act_dict))
    net.layer4[1].conv2.register_forward_hook(get_activation('conv_17', act_dict))

    example_data = example_data.to(device)
    example_out  = net(example_data)

    batch_list = np.concatenate((np.array([int(example_num / 10)] * 1),
                                 np.array([int(example_num / 2)] * 7),
                                 np.array([int(example_num)] * 10)))
    # network arch
    stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
    map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4]
    in_channel = [3, 64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512]

    pad = 1
    sc_list=[5,9,13]
    p1d = (1, 1, 1, 1)
    mat_final=[]
    mat_list=[]
    mat_sc_list=[]
    act_key = list(act_dict.keys())
    for i in range(len(stride_list)):
        if i==0:
            ksz = 3
        else:
            ksz = 3
        bsz=batch_list[i]
        st = stride_list[i]
        k=0
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        act = F.pad(act_dict[act_key[i]], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    k +=1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            act = act_dict[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        k +=1
            mat_sc_list.append(mat)

    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
    print('-'*30)
    return mat_final


def update_memory (model, mat_list, threshold, feature_list=[],):
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
            rank = min(rank, feature_list[i].shape[1])
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
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ## Load Mini-imagenet DATASET
    from dataloader import miniimagenet as data_loader
    dataloader = data_loader.DatasetGen(args, class_order=class_order2)
    taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

    acc_matrix=np.zeros((20,20))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([0.99] * 20)

        data = dataloader.get(k)
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x'].to(device)
        ytrain=data[k]['train']['y'].to(device)
        xvalid=data[k]['valid']['x'].to(device)
        yvalid=data[k]['valid']['y'].to(device)
        xtest =data[k]['test']['x'].to(device)
        ytest =data[k]['test']['y'].to(device)
        dst_train = xtrain
        dst_train.classes = range(5)
        dst_train.targets = data[k]['train']['y'].cpu()
        task_list.append(k)

        lr = args.lr
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)

        if task_id==0:
            model = ResNet_84x84(BasicBlock, [2, 2, 2, 2], num_features=args.num_features, backpack=args.backpack).to(device)

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
            mat_list = get_representation_matrix_ResNet18 (model, device, xtrain, ytrain, dst_train, k)
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
            mat_list = get_representation_matrix_ResNet18(model, device, xtrain, ytrain, dst_train, k)
            feature_list = update_memory (model, mat_list, threshold, feature_list)

        # save accuracy 
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x'].to(device)
            ytest =data[ii]['test']['y'] .to(device)
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
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()))
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index=[i for i in ["1", "2", "3", "4", "5", "6", "7",\
                                                   "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                                   "20"]],
                         columns=[i for i in ["1", "2", "3", "4", "5", "6", "7",\
                                              "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                              "20"]])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='mini imagenet')
    parser.add_argument('--batch_size_train', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--batch_size_test', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--num_features', type=int, default=64, metavar='N',
                        help='features of the base layer')
    parser.add_argument('--active_forget', default=0.99999, type=float,
                        help='proportion of active_forget')
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
    parser.add_argument('--fraction', default=0.01, type=float, help='fraction of data to be selected (default: 0.01)')
    parser.add_argument('--example_num', default=25, type=int, help='random selected samples (default: 25)')
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
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.01, help='learning rate for selection')
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="LazyGreedy",
                        help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")
    parser.add_argument('--model_name', type=str, default='ResNet_84x84', help='model')
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



