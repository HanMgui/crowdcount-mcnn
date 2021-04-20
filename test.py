import os
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path = '../data/original/shanghaitech/part_A_final/test_data/images/'
gt_path = '../data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'
# model_path = '../final_models/MCNNandMBv1_1_shtechA_1814.h5'
output_dir = '../output/'
# model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results.txt')

# output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)
model_path='../final_models/'
models_name=['MCNNandMBv1-1_shtechA_1814.h5','MCNNandMBv1-1_shtechA_2000.h5']
for i in range(len(models_name)):
    output_dir = '../output/'
    model_path = os.path.join('../final_models/',models_name[i])
    model_name = os.path.basename(model_path).split('.')[0]
    output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    net = CrowdCounter(model_name.split('_')[0])

    trained_model = os.path.join(model_path)

    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0
    maxone = -1.0
    maxones = '.'

    # #load test data
    # data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        if maxone < abs(gt_count-et_count):
            maxone = abs(gt_count-et_count)
            maxones = blob['fname'].split('.')[0]
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))
        if vis:
            utils.display_results(im_data, gt_data, density_map)
        if save_output:
            utils.save_density_map(np.concatenate((gt_data,np.ones((1,1,gt_data.shape[2],3)),density_map),axis=3), output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    print('\nMAE: %0.2f, MSE: %0.2f, max: %0.2f,num:%s' % (mae,mse,maxone,maxones))

    f = open(file_results, 'a')
    s=model_path+':MAE: %0.2f, MSE: %0.2f, max: %0.2f,num:%s\n' % (mae,mse,maxone,maxones)
    f.write(s)
    f.close()
    print('结束')