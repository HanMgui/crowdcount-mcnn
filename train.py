import os
from PIL import ImageGrab
import torch
import numpy as np
import sys
import shutil

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

# method = 'mcnn'
method = 'mcnnandmnv1'
dataset_name = 'shtechA'
output_dir = '../saved_models/'
netparams_dir = './saved_net/'

train_path = '../data/formatted_trainval/shanghaitech_part_A_patches_9/train'    #训练集
train_gt_path = '../data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'  # density
val_path = '../data/formatted_trainval/shanghaitech_part_A_patches_9/val'        #验证集
val_gt_path = '../data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'  # gt ground truth

# training configuration
start_step = 0
end_step = 2000
lr = 0.00001
momentum = 0.9
disp_interval = 1000
log_interval = 250

# Tensorboard  config
use_tensorboard = False
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False  # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard

# ------------
rand_seed = 64678
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

# load net
net = CrowdCounter()
network.weights_normal_init(net, dev=0.01)
net.cuda()
net.train()

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(netparams_dir):
    os.mkdir(netparams_dir)

# tensorboad 训练可视化
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = save_exp_name
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.sleepseconds(1)
print('dsa')
# t.tic()
ifpre_load = True
data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=ifpre_load)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=ifpre_load)
best_mae = sys.maxsize
best_mse = best_mae
best_model = -1
is_continue_train = True
if is_continue_train and os.path.exists(os.path.join(output_dir, 'xunliandata.hmg')):
    checkpoint = torch.load(os.path.join(output_dir, 'xunliandata.hmg'))
    train_loss = checkpoint['train_loss']
    step_cnt = checkpoint['step_cnt']
    start_step = checkpoint['epoch'] + 1
    data_loader = checkpoint['data_loader']
    data_loader_val = checkpoint['data_loader_val']
    best_mae = checkpoint['best_mae']
    best_mse = checkpoint['best_mse']
    best_model = checkpoint['best_model']
t.tic()
issuccss = False
try:
    for epoch in range(start_step, end_step + 1):
        duration = t.toc(average=False)
        t.tic()
        step = -1
        train_loss = 0
        for blob in data_loader:
            step = step + 1
            im_data = blob['data']  #图片
            gt_data = blob['gt_density']    #密度图
            density_map = net(im_data, gt_data) #计算预测值并使用gt_data构建net.loss
            loss = net.loss
            train_loss += loss.item()  # loss.data[0]#hmg
            step_cnt += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % disp_interval == 0:
                # duration = t.toc(average=False)
                # fps = step_cnt / duration
                gt_count = np.sum(gt_data)
                density_map = density_map.data.cpu().numpy()
                et_count = np.sum(density_map)
                utils.save_results(im_data, gt_data, density_map, output_dir)
                # log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f, duration: %4f' % (epoch,
                #     step, 1./fps, gt_count,et_count,duration)
                log_text = 'epoch: %4d, step %4d, Time: %4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % \
                           (epoch,      step,   duration,    gt_count,       et_count)
                log_print(log_text, color='green', attrs=['bold'])
                # re_cnt = True

            # if re_cnt:
            # t.tic()
            # re_cnt = False

        if epoch % 2 == 0:
            save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
            network.save_net(save_name, net)
            # calculate error on the validation dataset
            mae, mse = evaluate_model(save_name, data_loader_val)
            if mae < best_mae:
                best_mae = mae
                best_mse = mse
                best_model = '{}_{}_{}.h5'.format(method, dataset_name, epoch)
            log_text = 'EPOCH: %d, MAE: %.1f, MSE: %0.1f' % (epoch, mae, mse)
            log_print(log_text, color='green', attrs=['bold'])
            log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae, best_mse, best_model)
            log_print(log_text, color='green', attrs=['bold'])
            if use_tensorboard:
                exp.add_scalar_value('MAE', mae, step=epoch)
                exp.add_scalar_value('MSE', mse, step=epoch)
                exp.add_scalar_value('train_loss', train_loss / data_loader.get_num_samples(), step=epoch)
            # 我的hmg
            if is_continue_train and epoch % 10 == 0:
                torch.save({'train_loss': train_loss,
                            'step_cnt': step_cnt,
                            'epoch': epoch,
                            'data_loader': data_loader,
                            'data_loader_val': data_loader_val,
                            'best_mae': best_mae,
                            'best_mse': best_mse,
                            'best_model': best_model
                            }, os.path.join(output_dir, 'xunliandata.hmg'))
    issuccss = True
finally:
    if issuccss:
        torch.save({'net': net, 'best_model': best_model}, os.path.join(output_dir, 'net.hmg'))
        if os.path.exists(os.path.join(output_dir, 'xunliandata.hmg')):
            os.remove(os.path.join(output_dir, 'xunliandata.hmg'))
        # 复制最好的和最后的net
        shutil.copyfile(os.path.join(output_dir,best_model), os.path.join(netparams_dir,best_model))
        shutil.copyfile(os.path.join(output_dir,'{}_{}_{}.h5'.format(method, dataset_name, end_step)),
                        os.path.join(netparams_dir,'{}_{}_{}.h5'.format(method, dataset_name, end_step)))
    pic = ImageGrab.grab()
    pic.save(os.path.join(output_dir, 'picture.jpg'))
    print('120s 后关机')
    t.sleepseconds(120)
    os.system('shutdown /s /t 5')
