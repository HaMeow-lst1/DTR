from easydict import EasyDict as edict

def get_name_replace(dataset):    #path of dataset
    if dataset == 'KAIST':
        return ['lwir', 'visible']
    if dataset == 'FLIR':
        return ['PreviewData', 'RGB']

      


DTR = edict()
DTR.dataset = 'FLIR'   #KAIST or FLIR
DTR.base_dataset_path = '/home/lisongtao/datasets/DTR/'
DTR.name_replace_lst = get_name_replace(DTR.dataset)

DTR.epoch1 = 1
DTR.epoch2 = 2
DTR.epoch3 = 12


DTR.lambda1_visibleBig = 0.25
DTR.lambda1_visibleSmall = 0.25

DTR.lambda2_visibleSmall = 0.5

DTR.lambda3_visibleBig = 0.1

DTR.lambda_disentangle = 0.1
DTR.lambda_restore = 0.1