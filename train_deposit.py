# train a student network distilling from teacher

from email.policy import default
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam


from tqdm import tqdm
import argparse
import os
import logging
import numpy as np

from utils.utils import RunningAverage, set_logger, Params
from model import *
from data_loader import fetch_dataloader


# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='experiments/CIFAR10/kd_normal/cnn', type=str)
parser.add_argument('--teacher_resume', default=None, type=str,
                    help='If you specify the teacher resume here, we will use it instead of parameters from json file')
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[1], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--low_resume', default='/home/yejingwen/Nasty-Teacher/experiments/CIFAR10/deposit/original/lowbest_model.tar', type=str)
parser.add_argument('--up_resume', default='/home/yejingwen/Nasty-Teacher/experiments/CIFAR10/deposit/original/upbest_model.tar', type=str)
parser.add_argument('--dep_size', default=3, type=int)


args = parser.parse_args()

device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])

import os
print('GPU id {}'.format(device_ids))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[0])

class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self):
		super(AT, self).__init__()
		self.p = 2

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am
    

def at(x):
    att = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    att[att<0.005] = 0
    #print (att)
    return att


def at_loss(x, y):
    attx = at(x)
    atty = at(y)
    #print((attx).size())
    #print(attx,atty)
    #atty[atty<0.2] = 0
    return (attx - atty).pow(2).mean()

def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    """
    alpha = params.alpha
    T = params.temperature
    random_label = torch.randint_like(labels,10)
    split = params.dep_size
    #print(F.log_softmax(outputs/T, dim=1).shape)


    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[:,split:]/T, dim=1),
                             F.softmax(teacher_outputs[:,split:]/T, dim=1)) * (alpha * T * T) + \
              nn.CrossEntropyLoss()(outputs, random_label) * (1. - alpha)

    return KD_loss


# ************************** training function **************************
def train_epoch_kd(model, deposit, t_low, t_up, optim_tar, optim_dep, loss_fn_kd, data_loader, params):
    model.train()
    deposit.train()
    t_low.eval()
    t_up.eval()
    loss_avg = RunningAverage()
    #loss3_avg = RunningAverage()
    criterionAT = AT()
    #criterionAT=criterionAT.cuda()
    #deploss_avg = RunningAverage()
    split=params.dep_size
    with tqdm(total=len(data_loader)) as t:  # Use tqdm for progress bar
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            if params.cuda:
                train_batch = train_batch.cuda()  # (B,3,32,32)
                labels_batch = labels_batch.cuda()  # (B,)

            # compute model output and loss
            output_batch1 = model(train_batch)  # logit without SoftMax
            target_batch = t_up(output_batch1) # target net output
            


            output_batch = deposit(train_batch)  # logit without SoftMax
            # deposit_batch = t_up(output_batch) + target_batch # summation of two logitss
            deposit_batch = t_up(output_batch)
            deposit_batch[:,split:] = target_batch[:,split:]
            #deposit_batch

            # get one batch output from teacher_outputs list
            with torch.no_grad():
                output_teacher_batch1 = t_low(train_batch)   # logit without SoftMax
                output_teacher_batch = t_up(output_teacher_batch1)

            # CE(output, label) + KLdiv(output, teach_out)
            alpha = params.alpha
            T = params.temperature
            
            loss1 = loss_fn_kd(target_batch, labels_batch, output_teacher_batch, params)
            loss2 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(deposit_batch[:,:split]/T, dim=1),
                             F.softmax(output_teacher_batch[:,:split]/T, dim=1)) * (alpha * T * T) + \
                                 nn.CrossEntropyLoss()(deposit_batch, labels_batch)*(1. - alpha)
            
            #loss3 = criterionAT(output_batch1, output_teacher_batch1.detach())
            #loss3 = at_loss(output_batch1, output_teacher_batch1)
            loss = loss1 + loss2 #-100*loss3
            #loss = -loss3
            
            #print(loss)
            optim_tar.zero_grad()
            optim_dep.zero_grad()
            loss.backward()
            optim_tar.step()
            optim_dep.step()

            # update the average loss
            loss_avg.update(loss.item())
            #loss3_avg.update(loss3.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg(), loss_avg()


def evaluate(model, t_up, loss_fn, data_loader3, data_loader7, params):
    model.eval()
    t_up.eval()
    # summary for current eval loop
    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader3:
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)
            output_batch = t_up(output_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean1 = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}

    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader7:
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)
            output_batch = t_up(output_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean2 = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}

    return metrics_mean1, metrics_mean2

def evaluate_dep(model, deposit, t_up, loss_fn, data_loader3, data_loader7, params):
    model.eval()
    deposit.eval()
    t_up.eval()
    # summary for current eval loop
    summ = []
    split = params.dep_size
    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader3:
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            target_batch = model(data_batch)
            target_batch = t_up(target_batch)
            output_batch = deposit(data_batch)
            output_batch = t_up(output_batch)
            output_batch[:,split:] = target_batch[:,split:]

            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean1 = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}

    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader7:
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            target_batch = model(data_batch)
            target_batch = t_up(target_batch)
            output_batch = deposit(data_batch)
            output_batch = t_up(output_batch) + target_batch
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean2 = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}

    return metrics_mean1, metrics_mean2

def train_and_eval_kd(model, deposit, t_low, t_up, optim_tar, optim_dep, loss_fn, train_loader, dev_loaderdep, dev_loaderpre, params):
    tarbest_val_acc = -1
    depbest_val_acc = -1
    tarbest_epo = -1
    depbest_epo = -1
    lr = params.learning_rate

    for epoch in range(params.num_epochs):
        # LR schedule *****************
        lr = adjust_learning_rate(optim_tar, epoch, lr, params)
        lr = adjust_learning_rate(optim_dep, epoch, lr, params)

        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info('Learning Rate {}'.format(lr))

        # ********************* one full pass over the training set *********************
        train_loss,loss3 = train_epoch_kd(model, deposit, t_low, t_up, optim_tar, optim_dep, loss_fn, train_loader, params)
        logging.info("- Train loss : {:05.3f}".format(train_loss))
        logging.info("- ATT loss : {:05.3f}".format(loss3))

        # ********************* Evaluate for one epoch on validation set on TargetNet *********************
        targetval_metrics1, targetval_metrics2 = evaluate(model, t_up, nn.CrossEntropyLoss(), dev_loaderdep, dev_loaderpre, params)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in targetval_metrics1.items())
        logging.info("- Target Eval metrics-dep : " + metrics_string)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in targetval_metrics2.items())
        logging.info("- Target Eval metrics-pre : " + metrics_string)

        # save model
        save_name = os.path.join(args.save_path, 'targetlast_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim_tar.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
        val_acc = targetval_metrics2['acc'] # get the best accuracy on the preservation set
        if val_acc >= tarbest_val_acc:
            tarbest_epo = epoch + 1
            tarbest_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'targetbest_model.tar')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim_tar.state_dict()},
                save_name)

        logging.info('- So far Target best epoch: {}, best acc: {:05.3f}'.format(tarbest_epo, tarbest_val_acc))

         # ********************* Evaluate for one epoch on validation set on depositNet *********************
        depositval_metrics1, depositval_metrics2 = evaluate_dep(model, deposit, t_up, nn.CrossEntropyLoss(), dev_loaderdep, dev_loaderpre, params)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in depositval_metrics1.items())
        logging.info("- Deposit Eval metrics-dep : " + metrics_string)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in depositval_metrics2.items())
        logging.info("- Deposit Eval metrics-pre : " + metrics_string)

        # save model
        save_name = os.path.join(args.save_path, 'last_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': deposit.state_dict(), 'optim_dict': optim_dep.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
        val_acc = depositval_metrics1['acc'] # get the best accuracy on the deposit set
        if val_acc >= depbest_val_acc:
            depbest_epo = epoch + 1
            depbest_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'best_model.tar')
            torch.save({
                'epoch': epoch + 1, 'state_dict': deposit.state_dict(), 'optim_dict': optim_dep.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(depbest_epo, depbest_val_acc))



def adjust_learning_rate(opt, epoch, lr, params):
    if epoch in params.schedule:
        lr = lr * params.gamma
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return lr


if __name__ == "__main__":
    # ************************** set log **************************
    set_logger(os.path.join(args.save_path, 'training.log'))

    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.cuda = torch.cuda.is_available() # use GPU if available

    for k, v in params.__dict__.items():
        logging.info('{}:{}'.format(k, v))

    # ########################################## Dataset ##########################################  
    params.dataset = 'cifar10'
    trainloader = fetch_dataloader('train', params)
    devloaderpre = fetch_dataloader('pre', params)
    devloaderdep = fetch_dataloader('dep', params)

    # ############################################ Model ############################################
    if params.dataset == 'cifar10':
        num_class = 10
    elif params.dataset == 'cifar100':
        num_class = 100
    elif params.dataset == 'tiny_imagenet':
        num_class = 200
    else:
        num_class = 10

    logging.info('Number of class: ' + str(num_class))

    # ############################### Student Model ###############################
    logging.info('Create Student Model --- ' + params.model_name)

    # ResNet 18 / 34 / 50 ****************************************
    if params.model_name == 'resnet18':
        #model = ResNet18(num_class=num_class)
        model_deposit = ResNet18low(num_class=num_class)
        model_target = ResNet18low(num_class=num_class)
    elif params.model_name == 'resnet34':
        model = ResNet34(num_class=num_class)
    elif params.model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif params.model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)
    elif params.model_name.startswith('preresnet44'):
        model = PreResNet(depth=44, num_classes=num_class)
    elif params.model_name.startswith('preresnet56'):
        model = PreResNet(depth=56, num_classes=num_class)
    elif params.model_name.startswith('preresnet110'):
        model = PreResNet(depth=110, num_classes=num_class)


    # DenseNet *********************************************
    elif params.model_name == 'densenet121':
        model = densenet121(num_class=num_class)
    elif params.model_name == 'densenet161':
        model = densenet161(num_class=num_class)
    elif params.model_name == 'densenet169':
        model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.model_name == 'resnext29':
        model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.model_name == 'mobilenetv2':
        model = MobileNetV2(class_num=num_class)

    elif params.model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.model_name == 'net':
        model = Net(num_class, params)

    elif params.model_name == 'mlp':
        model = MLP(num_class=num_class)

    else:
        model = None
        print('Not support for model ' + str(params.model_name))
        exit()

    # ############################### Teacher Model ###############################
    logging.info('Create Teacher Model --- ' + params.teacher_model)
    # ResNet 18 / 34 / 50 ****************************************
    if params.teacher_model == 'resnet18':
        #teacher_model = ResNet18(num_class=num_class)
        teacher_low = ResNet18low(num_class=num_class)
        teacher_up = ResNet18up(num_class=num_class)
    elif params.teacher_model == 'resnet34':
        teacher_model = ResNet34(num_class=num_class)
    elif params.teacher_model == 'resnet50':
        teacher_model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.teacher_model.startswith('preresnet20'):
        teacher_model = PreResNet(depth=20)
    elif params.teacher_model.startswith('preresnet32'):
        teacher_model = PreResNet(depth=32)
    elif params.teacher_model.startswith('preresnet56'):
        teacher_model = PreResNet(depth=56)
    elif params.teacher_model.startswith('preresnet110'):
        teacher_model = PreResNet(depth=110)

    # DenseNet *********************************************
    elif params.teacher_model == 'densenet121':
        teacher_model = densenet121(num_class=num_class)
    elif params.teacher_model == 'densenet161':
        teacher_model = densenet161(num_class=num_class)
    elif params.teacher_model == 'densenet169':
        teacher_model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.teacher_model == 'resnext29':
        teacher_model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.teacher_model == 'mobilenetv2':
        teacher_model = MobileNetV2(class_num=num_class)

    elif params.teacher_model == 'shufflenetv2':
        teacher_model = shufflenetv2(class_num=num_class)

    elif params.teacher_model == 'net':
        teacher_model = Net(num_class, args)

    elif params.teacher_model == 'mlp':
        teacher_model = MLP(num_class=num_class)

    else:
        teacher_model = None
        exit()

    if params.cuda:
        model_target = model_target.cuda()
        model_deposit = model_deposit.cuda()
        teacher_low = teacher_low.cuda()
        teacher_up = teacher_up.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        teacher_model = nn.DataParallel(teacher_model, device_ids=device_ids)

    # checkpoint ********************************
    if args.up_resume:
        logging.info('- Load checkpoint model from {}'.format(args.resume))
        checkpoint = torch.load(args.low_resume,map_location='cuda:1')
        teacher_low.load_state_dict(checkpoint['state_dict'])
        model_target.load_state_dict(checkpoint['state_dict'])
        model_deposit.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(args.up_resume,map_location='cuda:1')
        teacher_up.load_state_dict(checkpoint['state_dict'])
        #checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        #model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('- Train from scratch ')


    # ############################### Optimizer ###############################
    if params.model_name == 'net' or params.model_name == 'mlp':
        optimizer = Adam(model.parameters(), lr=params.learning_rate)
        logging.info('Optimizer: Adam')
    else:
        optimizer_tar = SGD(model_target.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
        optimizer_dep = SGD(model_deposit.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)

        logging.info('Optimizer: SGD')

    # ************************** LOSS **************************
    criterion = loss_fn_kd

    # ************************** Teacher ACC **************************
    logging.info("- Teacher Model Evaluation ....")
    #val_metrics = evaluate(teacher_model, nn.CrossEntropyLoss(), devloader, params)  # {'acc':acc, 'loss':loss}
    val_metrics1, val_metrics2= evaluate(teacher_low, teacher_up, nn.CrossEntropyLoss(), devloaderdep, devloaderpre, params)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics1.items())
    logging.info("- Teacher Model Eval metrics3 : " + metrics_string)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics2.items())
    logging.info("- Teacher Model Eval metrics7 : " + metrics_string)

    # ************************** train and evaluate **************************
    train_and_eval_kd(model_target, model_deposit, teacher_low, teacher_up, optimizer_tar, optimizer_dep, criterion, trainloader, devloaderdep, devloaderpre, params)


