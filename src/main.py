import cv2
import os
import datetime
import numpy as np
import argparse
from models.Sal_based_Attention_module import  Sal_based_Attention_module
from models.Sal_global_Attention import  Sal_global_Attention
from models.SalGAN360 import SalGAN360
import pickle
import torch
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
from torch import nn , nn.BCELoss
from loss import NSS ,CC ,KLD 
from torch.utils import data
from data_loader import Static_dataset

def get_parser():
    parser.add_argument('-use_gpu', dest='use_gpu', default='parallel', help="If you are using cuda set to 'gpu'. If you want to use the DataParallel pytorch module set to 'parallel'. Otherwise, set to 'cpu'.")
    parser.add_argument('-src', dest='src', default="../data", help="Add root path to your dataset.")
    # Dataset
    parser.add_argument('-dataset', dest='dataset', default='DHF1K', help="Name of the dataset to train model with. These can be 'DHF1K', 'Hollywood-2', 'UCF-sports'. If you wish to use your own dataset input 'other'.")
    parser.add_argument('-end', dest='end', default=,type=int, help='Define at which you would wish to end .')
    # Training
    parser.add_argument('-lr', dest='lr', default=0.000001, type=float, help='Learning rate used when optimizing the parameters. ')
    parser.add_argument('-epochs', dest='epochs', default=0, type=int, help='Number of epochs to run. ')
    parser.add_argument('-val_perc', dest='val_perc', default=0.1, type=float, help='Percentage to use as validation set for the loss metric.')
    # Model
    parser.add_argument('-pt_model', dest='pt_model', default='', help="Input path to a pretrained model.")
    parser.add_argument('-new_model', dest='new_model', default='', help="Input name to use on newly trained model")

    return parser
def main(args, params ):

    # Data Loading


    if args.dataset == "Static_dataset":
        print("Commencing training on dataset {}".format(args.dataset))
        train_set = Static_dataset(
            root_path = args.src,
            load_gt = True,
            number_of_frames = int(args.end),
            resolution = (640, 320),
            val_perc = args.val_perc,
            split = "train")
        print("Size of train set is {}".format(len(train_set)))
        train_loader = data.DataLoader(train_set, **params)

        if args.val_perc > 0:
            val_set = Static_dataset(
                root_path = args.src,
                load_gt = True,
                number_of_frames = int(args.end),
                resolution = (640, 320),
                val_perc = args.val_perc,
                split = "validation")
            print("Size of validation set is {}".format(len(val_set)))
            val_loader = data.DataLoader(val_set, **params)
    else:
        print('Your dataset was not recognized. Check the name and try again!!!.')
        exit()


    # Models

    if 'Sal_based_Attention_module' in args.pt_model:
        model = Sal_based_Attention_module()
        print("Initialized {}".format(args.new_model)) 

    elif 'Sal_global_Attention' in args.pt_model:
        model = Sal_based_Attention_module()
        print("Initialized {}".format(args.new_model))

    else:
        print("Your model was not recognized, check the name of the model and try again.")
        exit()

    criterion = BCELoss() + 0.1*NSS() + 0.1*KLD() + 0.1*CC() 

    optimizer = torch.optim.Adam([
        {'params':model.parameters() , 'lr': args.lr}])


    checkpoint = load_weights(model, args.pt_model)
    model.load_state_dict(checkpoint, strict=False)
    start_epoch = torch.load(args.pt_model, map_location='cpu')['epoch']
    

    print("Model loaded, start training from epoch {}".format(start_epoch))

    dtype = torch.FloatTensor

    if args.use_gpu == 'parallel' or args.use_gpu == 'gpu':
        assert torch.cuda.is_available(), \
            "CUDA is not available in your machine"

        if args.use_gpu == 'parallel':
            model = nn.DataParallel(model).cuda()

        elif args.use_gpu == 'gpu':
            model = model.cuda()

        dtype = torch.cuda.FloatTensor
        cudnn.benchmark = True  

        criterion = criterion.cuda()

    #Traning #
    
    starting_time = datetime.datetime.now().replace(microsecond=0)
    print("Training started at : {}".format(starting_time))

    for epoch in range(start_epoch, args.epochs+1):

        try:
            # train for one epoch
            train_loss, optimizer = train(train_loader, model, criterion, optimizer, epoch, args.use_gpu , dtype)

            print("Epoch {}/{} done with train loss {}\n".format(epoch, args.epochs, train_loss))

            if args.val_perc > 0:
                print("Running validation..")
                val_loss = validate(val_loader, model, criterion, epoch, dtype)
                print("Validation loss: {}".format(val_loss))

            if epoch % plot_every == 0:
                train_losses.append(train_loss.cpu())
                if args.val_perc > 0:
                    val_losses.append(val_loss.cpu())

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.cpu().state_dict(),
                'optimizer' : optimizer.state_dict()
                }, args.new_model+".pt")

            if args.use_gpu == 'parallel':
                model = nn.DataParallel(model).cuda()
            elif args.use_gpu == 'gpu':
                model = model.cuda()


        except RuntimeError:
            print("A memory error was encountered. Further training aborted.")
            epoch = epoch - 1
            break

    print("Training of {} started at {} and finished at : {} \n Now saving..".format(args.new_model, starting_time, datetime.datetime.now().replace(microsecond=0)))

    # Saving #


    if args.val_perc > 0:
        to_plot = {
            'epoch_ticks': list(range(start_epoch, args.epochs+1, plot_every)),
            'train_losses': train_losses,
            'val_losses': val_losses
            }
        with open('to_plot.pkl', 'wb') as handle:
            pickle.dump(to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_weights(model, pt_model, device='cpu'):
    # Load stored model:
    temp = torch.load(pt_model, map_location=device)['state_dict']
    # Because of dataparallel there is contradiction in the name of the keys so we need to remove part of the string in the keys:.
    from collections import OrderedDict
    checkpoint = OrderedDict()
    for key in temp.keys():
        new_key = key.replace("module.","")
        checkpoint[new_key]=temp[key]

    return checkpoint

def train(train_loader, model, criterion, optimizer, epoch, dtype):


    # Switch to train mode
    model.train()

    print("Now commencing epoch {}".format(epoch))

    losses = []
    for  j,images in enumerate(train_loader):
     
        start = datetime.datetime.now().replace(microsecond=0)
        loss = 0
        for i in range(images.size()[0]):
            frame , gtruth , fixation = images[i]
            saliency_map , attention_map = model.forward(frame)
            saliency_map = saliency_map.squeeze(0)
            attention_map = attention_map.squeeze(0)
            if saliency_map.size() != gtruth.size():
                print(saliency_map.size())
                print(gtruth.size())
                a, b, c, _ = saliency_map.size()
                saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3)
            loss = loss + criterion(saliency_map, gtruth) +criterion(attention_map, fixation)

        loss.backward()
        optimizer.step()

        if i%100==0 :

            post_process_saliency_map = (saliency_map-torch.min(saliency_map))/(torch.max(saliency_map)-torch.min(saliency_map))
            utils.save_image(post_process_saliency_map, "./log-map/smap{}_epoch{}.png".format(i, epoch))


        end = datetime.datetime.now().replace(microsecond=0)
        print('Epoch: {}\Batch: {}\t Training Loss: {}\t Time elapsed: {}\t'.format(epoch, j, loss.data, end-start))
        losses.append(loss.data)

    return (mean(losses), n_iter, optimizer)

def validate(val_loader, model, criterion, epoch ):

    # switch to evaluate mode
    model.eval()

    losses = []
    print("Now running validation..")
    with torch.no_grad():
        for  j,images in enumerate(val_loader):
            loss = 0
            for i in range(images.size()[0]):
                frame , gtruth , _ = images[i]
                saliency_map , _ = model.forward(frame)
                saliency_map = saliency_map.squeeze(0)
                
                if saliency_map.size() != gtruth.size():
                    print(saliency_map.size())
                    print(gtruth.size())
                    a, b, c, _ = saliency_map.size()
                    saliency_map = torch.cat([saliency_map, torch.zeros(a, b, c, 1).cuda()], 3)
                loss = loss + criterion(saliency_map, gtruth)

            losses.append(loss.data)

    return(mean(losses))

if __name__ == '__main__':
    parser = get_parser
    args = parser.parse_args()
    main(args, params = {'batch_size': 10,'num_workers': 4,'pin_memory': True})
