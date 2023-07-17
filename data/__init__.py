"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
#
# def CreateDataLoader(opt):
#     data_loader = CustomDatasetDataLoader()
#     print(data_loader.name())
#     data_loader.initialize(opt)
#     return data_loader
#
#
# def CreateDataset(opt):
#     #data directory
#     target_file=opt.dataroot+'/'+opt.phase+'/data.mat'
#     f = h5py.File(target_file,'r')
#     slices=np.array(f['data_x']).shape[3]/2
#     samples=range(np.array(f['data_y']).shape[2])
#     #Selecting neighbouring slices based on the inputs
#     if opt.direction=='AtoB':
#         data_x=np.array(f['data_x'])[:,:,:,:]
#         print(data_x.shape)
#         data_y=np.array(f['data_y'])[:,:,:,:]
#     else:
#         data_y=np.array(f['data_y'])[:,:,:,:]
#         data_x=np.array(f['data_x'])[:,:,:,:]
#     #Shuffle slices in data_y for the cGAN case (incase the input data is registered)
#     if opt.dataset_mode == 'unaligned_mat':
#         if opt.isTrain:
#             print("Training phase")
#             random.shuffle(samples)
#         else:
#             print("Testing phase")
#         data_y=data_y[:,:,samples,:]
#     data_x=np.transpose(data_x,(3,2,0,1))
#     data_y=np.transpose(data_y,(3,2,0,1))
#     #Ensure that there is no value less than 0
#     data_x[data_x<0]=0
#     data_y[data_y<0]=0
#     dataset=[]
#     #making range of each image -1 to 1 and converting to torch tensor
#     for train_sample in range(data_x.shape[1]):
#         data_x[:,train_sample,:,:]=(data_x[:,train_sample,:,:]-0.5)/0.5
#         data_y[:,train_sample,:,:]=(data_y[:,train_sample,:,:]-0.5)/0.5
#         dataset.append({'A': torch.from_numpy(data_x[:,train_sample,:,:]), 'B':torch.from_numpy(data_y[:,train_sample,:,:]),
#         'A_paths':opt.dataroot, 'B_paths':opt.dataroot})
#     #If number of samples in data_x and data_y are different (for unregistered images) make them same
#     if data_x.shape[1]!=data_y.shape[1]:
#         for train_sample in range(max(data_x.shape[1],data_y.shape[1])):
#             if data_x.shape[1]>=data_y.shape[1] and train_sample>(data_y.shape[1]-1):
#                 data_x[:,train_sample,:,:]=(data_x[:,train_sample,:,:]-0.5)/0.5
#                 dataset.append({'A': torch.from_numpy(data_x[:,train_sample,:,:]), 'A_paths':opt.dataroot})
#             elif data_y.shape[1]>data_x.shape[1] and train_sample>(data_x.shape[1]-1):
#                 data_y[:,train_sample,:,:]=(data_y[:,train_sample,:,:]-0.5)/0.5
#                 dataset.append({ 'B':torch.from_numpy(data_y[:,train_sample,:,:]), 'B_paths':opt.dataroot})
#             else:
#                 data_x[:,train_sample,:,:]=(data_x[:,train_sample,:,:]-0.5)/0.5
#                 data_y[:,train_sample,:,:]=(data_y[:,train_sample,:,:]-0.5)/0.5
#                 dataset.append({'A': torch.from_numpy(data_x[:,train_sample,:,:]), 'B':torch.from_numpy(data_y[:,train_sample,:,:]),
#                 'A_paths':opt.dataroot, 'B_paths':opt.dataroot})
#     return dataset
#
# class CustomDatasetDataLoader(BaseDataLoader):
#     def name(self):
#         return 'CustomDatasetDataLoader'
#
#     def initialize(self, opt):
#         BaseDataLoader.initialize(self, opt)
#         self.dataset = CreateDataset(opt)
#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset,  #  数据加载
#             batch_size=opt.batch_size,  # 设置批处理大小
#             shuffle=not opt.serial_batches,  # 是否随机洗牌
#             num_workers=int(opt.num_threads))  # 是否进行多进程加载数据设置
#
#
#     def load_data(self):
#         return self
#
#
#     def __len__(self):
#         return min(len(self.dataset), self.opt.max_dataset_size)
#
#     def __iter__(self):
#         for i, data in enumerate(self.dataloader):
#             if i >= self.opt.max_dataset_size:
#                 break
#             yield data



# 以下原
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
#
#
def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"  # data.aligned_dataset
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'  #  dataset_name为aligned,变为aligneddataset
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    # print('刚读入时，Min: %.3f, Max: %.3f' % (dataset.min(), dataset.max()))
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)  # unaligned/ aligned
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,  #  数据加载
            batch_size=opt.batch_size,  # 设置批处理大小
            shuffle=not opt.serial_batches,  # 是否随机洗牌
            num_workers=int(opt.num_threads))  # 是否进行多进程加载数据设置

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
