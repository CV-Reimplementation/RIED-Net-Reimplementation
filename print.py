import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # opt = TrainOptions().parse()   # get training options
    # print(opt)
    # print("opt——mode", opt.dataset_mode)

    opt = TestOptions().parse()  # get test options

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of test images = %d' % dataset_size)