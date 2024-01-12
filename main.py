import time
from test import test
from train import train
from preProcess import pre_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # train and test setting
    # Grocery_and_Gourmet_Food_5
    # Office_Products_5
    # Sports_and_Outdoors_5
    # Musical_Instruments_5
    # Video_Games_5
    filename = "./dataset/Sports_and_Outdoors_5.json"
    dataset = filename.split('/')[-1].split('.')[0]
    pth_path = './checkpoints/DAML_{}_default.pth'.format(dataset)
    kwargs = {
        'dataset': dataset,
        'model': 'DAML',
        'num_fea': 2,
        'batch_size': 128,
        'gpu_id': 0,
        'filters_num': 10,
        'output': 'nfm',
        'num_epochs': 100,
        'self_att': True,
        'pth_path': pth_path,
        'lr': 1e-4,
        'optimizer':'ADAM'#SGD
    }

    # pre_data.pre_process(filename)
    # train(kwargs)
    test(kwargs)

    # 统计卷积核的个数对时间的影响
    # used_time_list = []
    # for i in range(1, 11):
    #     kwargs['filters_num'] = i * 10
    #     now = time.time()
    #     train(kwargs)
    #     #释放空间
    #
    #     print("filters_num: ", i * 10, "time: ", time.time() - now)
    #     used_time_list.append(time.time() - now)
    # # 绘制，时间-卷积核个数曲线
    # print(used_time_list)
    # plt.plot([i * 10 for i in range(1, 11)], used_time_list, 'r-')
    # plt.show()
    pass
