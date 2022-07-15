"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from torch.autograd import Variable
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torchvision.utils import save_image

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)  # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    results = []

    for i, data in enumerate(dataset):

        img_path = data['A_paths'][0]  # '/home/jijang/projects/Drone/DCLGAN/datasets/drone_smoothing/testA/0006_00026.jpg'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        crop = []
        position = []
        batch_count = 0

        result_img = np.zeros_like(img)  # ndarray (2160, 2840, 3)
        voting_mask = np.zeros_like(img)  # ndarray (2160, 2840, 3)

        cnt = 0

        for top in tqdm(range(0, img.shape[0], opt.stride)):  # moving by stride (row)
            for left in range(0, img.shape[1], opt.stride):  # moving by stride (col)
                piece = np.zeros([opt.piece, opt.piece, 3], np.float32)  # ndarray (h, w, c)
                temp = img[top:top + opt.piece, left:left + opt.piece, :]  # ndarray (h, w, c)
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                crop.append(piece)
                position.append([top, left])

                crop = np.array(crop).astype('float32')  # ndarray (b, h, w, c)
                crop = torch.from_numpy(crop)  # tensor (b, h, w, c)
                crop = crop.permute(0, 3, 1, 2)  # tensor (b, c, h, w)
                # .unsqueeze(0) => make batch 1
                crop = crop / 255.0

                data['A'] = crop
                data['B'] = crop
                crop = []

                try:
                    A_directory = opt.dataroot + '/crop_testA'
                    # if not os.path.exists(A_directory):
                    #    os.makedirs(A_directory)
                except OSError:
                    print('Error: Creating directory. ' + A_directory)

                try:
                    B_directory = opt.dataroot + '/crop_testB'
                    # if not os.path.exists(B_directory):
                    #    os.makedirs(B_directory)
                except OSError:
                    print('Error: Creating directory. ' + B_directory)

                data['A_paths'] = A_directory + '/' + str(cnt) + '_' + img_path.split('/')[-1]
                data['B_paths'] = B_directory + '/' + str(cnt) + '_' + img_path.split('/')[-1]

                model.data_dependent_initialize(data)
                model.setup(opt)  # regular setup: load and print networks; create schedulers
                # model.parallelize()

                if opt.eval:
                    model.eval()
                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break
                model.set_input(data)  # unpack data from data loader
                model.test()
                visuals = model.get_current_visuals()  # get image results

                # save_image(visuals['fake_A'], data['A_paths'], normalize=True)
                # save_image(visuals['fake_B'], data['B_paths'], normalize=True)

                for num, (t, l) in enumerate(position):
                    piece = visuals['fake_B'][num]
                    piece = piece.permute(1, 2, 0)
                    h, w, c = result_img[t:t + opt.piece, l:l + opt.piece, :].shape
                    # result_img = torch.Tensor(result_img).cuda()
                    piece = piece.cpu().detach().numpy()
                    result_img = result_img.astype('float32')
                    result_img[t:t + opt.piece, l:l + opt.piece, :] += piece[:h, :w, :]
                    voting_mask[t:t + opt.piece, l:l + opt.piece, :] += 1
                position = []

                cnt += 1

        result_img = result_img / voting_mask
        # result_img = result_img.astype(np.uint8)
        # results.append(result_img)
        results = np.array(result_img).astype('float32')
        results = torch.from_numpy(results)
        results = results.permute(2, 0, 1).unsqueeze(0)  # tensor (b, c, h, w)
        # print(results)
        # save image to your output folder dir
        save_image(results, '/home/jijang/projects/Drone/DCLGAN/smoothing/winter/'+img_path.split('/')[-1], nrow=5, normalize=True)
