from tqdm import tqdm
from visdom import Visdom

from data.build_data import build_semi_dataloader, batch_transform
from model import UNet, DeepLabV3p
from myutils.utils import *
from torch.cuda.amp import GradScaler, autocast


def main_CASSF(args, bviz=False):

    # Define seed & device
    fix_seed_for_reproducibility(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for visualization
    if bviz:
        viz = Visdom()

    # Get Dataloader for Labeled/Unlabeled
    train_l_loader, train_u_loader = build_semi_dataloader(args)
    # Number of batches
    len_l = len(train_l_loader)
    len_u = len(train_u_loader)
    train_epoch = args.iter_per_epoch

    # Get CNN model
    if args.model == 'UNet':
        net = UNet(in_channels=args.in_channel, out_channels=args.class_num, proj_dim=args.out_dim).to(device)
    elif args.model == 'DeepLabv3p':
        net = DeepLabV3p(args.backbone, in_channels=args.in_channel, n_class=args.class_num, pretrained=True,
                         out_dim=args.out_dim).to(device)
    else:
        raise NotImplementedError

    ema = EMA(net, args.total_epoch * train_epoch)  # Mean-teacher model

    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                                nesterov=True)
    scheduler = PolyLR(optimizer, args.total_epoch * train_epoch, power=0.9)

    # Arrays to hold the best/per-epoch metrics
    epoch_cost = np.zeros((args.total_epoch, 4))
    cost = np.zeros(4)  # Temporarily hold for losses

    ''' start training'''
    iteration = 0
    for epoch in range(args.total_epoch):
        torch.cuda.empty_cache()  # empty cuda cache

        # Set model status for training
        net.train()  # student network
        ema.model.train()  # teacher network

        '''train'''
        # Traverse data: Labeled(1 batch) + Unlabeled(1 batch)
        scaler = GradScaler()
        for _ in tqdm(range(train_epoch)):
            torch.cuda.empty_cache()

            # Re-iterate dl/du to fit the size of train_epoch
            if iteration % len_l == 0:
                train_l_dataset = iter(train_l_loader)

            if iteration % len_u == 0:
                train_u_dataset = iter(train_u_loader)

            # Get labeled data and label
            train_l_data, train_l_label = train_l_dataset.__next__()
            train_l_data, train_l_label = train_l_data.to(device), train_l_label.to(device, dtype=torch.long)

            # Get unlabeled data
            train_u_data, _ = train_u_dataset.__next__()
            train_u_data = train_u_data.to(device)

            # Generate Pseudo labels using Teacher Network
            with torch.no_grad():
                _, pred_u_raw = ema.model(train_u_data)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_raw, dim=1), dim=1)

                # Apply random augmentation for unlabeled data
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    batch_transform(train_u_data, pseudo_labels, pseudo_logits)

            with autocast():
                # Generate Encoding Features & Predictions using Student Network
                feat_l, pred_l = net(train_l_data)
                feat_u, pred_u = net(train_u_aug_data)

                '''loss1: supervised-learning loss'''
                sup_loss = compute_supervised_loss(pred_l, train_l_label)
                '''loss2: unsupervised-learning loss'''
                if args.apply_unsup:
                    unsup_loss, mask_u = compute_unsupervised_loss(pred_u, train_u_aug_label, train_u_aug_logits,
                                                                threshold=args.unsup_threshold,
                                                                mode=args.unsupervised_mode)
                else:
                    unsup_loss = torch.tensor(0.0)

                '''loss3: contrastive loss'''
                # compute CC loss
                if args.apply_cc:
                    feat_all = torch.cat((feat_l, feat_u))
                    with torch.no_grad():
                        # Generate mask and one-hot label
                        mask_l = (train_l_label.unsqueeze(1) >= 0).float()

                        if not args.apply_unsup:
                            mask_u = torch.zeros(train_u_aug_logits.shape).to(device)

                        mask_all = torch.cat((mask_l, mask_u.unsqueeze(1)))

                        label_l = label_onehot(train_l_label, args.class_num)
                        label_u = label_onehot(train_u_aug_label, args.class_num)
                        label_all = torch.cat((label_l, label_u))

                        prob_l = torch.softmax(pred_l, dim=1)
                        prob_u = torch.softmax(pred_u, dim=1)
                        prob_all = torch.cat((prob_l, prob_u))

                    cc_loss = compute_caco_loss(feat_all, label_all, mask_all, prob_all, args.strong_threshold,
                                                  args.temperature)
                else:
                    cc_loss = torch.tensor(0.0)

            # final loss
            loss = sup_loss + unsup_loss + cc_loss

            # Optimize Student Network
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update Teacher Network
            ema.update(net)

            # Record cost
            cost[0] = loss.item()
            cost[1] = sup_loss.item()
            if args.apply_unsup:
                cost[2] = unsup_loss.item()
            if args.apply_cc:
                cost[3] = cc_loss.item()
            epoch_cost[epoch] += cost / train_epoch
            iteration += 1
            scheduler.step()

            # visualization of loss curve & batch data
            if bviz:
                # Draw loss curves
                viz.line([[sup_loss.item(), unsup_loss.item(), cc_loss.item()]], [iteration], win='train_loss',
                         update='append',
                         opts=dict(legend=['suploss', 'unsuploss', 'ccloss']))

                # convert greyscale from range(0,class_num) to (0,255) for display
                n_class = args.class_num
                labels_disp = train_l_label / n_class * 255
                labelsu_disp = train_u_aug_label / n_class * 255
                _, out_u = torch.max(torch.softmax(pred_u, dim=1), dim=1)
                predsu_disp = out_u / n_class * 255
                _, out_l = torch.max(torch.softmax(pred_l, dim=1), dim=1)
                preds_disp = out_l / n_class * 255

                # Display images
                patch_size = args.patch_size
                viz.images(train_l_data[:, :3].view(-1, 3, patch_size, patch_size), win='myimg', opts=dict(title='img'))
                viz.images(train_l_data[:, -1].view(-1, 1, patch_size, patch_size), win='mydsm', opts=dict(title='dsm'))
                viz.images(labels_disp.view(-1, 1, patch_size, patch_size), win='mylabel', opts=dict(title='lab'))
                viz.images(preds_disp.view(-1, 1, patch_size, patch_size), win='mypred', opts=dict(title='pred'))

                viz.images(train_u_data[:, :3].view(-1, 3, patch_size, patch_size), win='myulimg_ori',
                           opts=dict(title='uimg_ori'))
                viz.images(train_u_aug_data[:, :3].view(-1, 3, patch_size, patch_size), win='myulimg',
                           opts=dict(title='uimg'))
                viz.images(labelsu_disp.view(-1, 1, patch_size, patch_size), win='myullabel',
                           opts=dict(title='ulab'))
                viz.images(predsu_disp.view(-1, 1, patch_size, patch_size), win='myulpred',
                           opts=dict(title='upred'))

        # Saving models & messages
        if (epoch + 1) == args.total_epoch:
            torch.save(ema.model.state_dict(), 'logging/{}/{}_label{}_semi_{}_ema_epoch{}.pth'
                       .format( args.dataset_name,args.dataset_name, args.per_lab, args.model, epoch + 1))
            torch.save(net.state_dict(), 'logging/{}/{}_label{}_semi_{}_net_epoch{}.pth'
                       .format(args.dataset_name,args.dataset_name, args.per_lab, args.model, epoch + 1))

        '''Output messages'''
        # printing
        print('EPOCH: {:04d} ITER: {:04d}'.format(epoch + 1, iteration))
        print('TRAIN [Loss | sup_loss | unsup_loss | cc_loss]: {:.4f} {:.4f} {:.4f} {:.4f}'
            .format(epoch_cost[epoch][0], epoch_cost[epoch][1], epoch_cost[epoch][2], epoch_cost[epoch][3]))


def get_args():
    """
      Arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='A Class-aware Semi-supervised Framework for Semantic Segmentation of High-resolution Remote Sensing Imagery')

    # DATASETS
    parser.add_argument('-dataset_name', default='Vaihingen', help='dataset name',
                        choices=['Potsdam', 'Vaihingen', 'LoveDA-Urban', 'LoveDA-Rural'])
    parser.add_argument('-train_data', default='data/Vaihingen/train', help='root folder of training data')
    parser.add_argument('-train_set', default='data/Vaihingen/splits/train_files.txt', help='txt file trainset')
    parser.add_argument('-lab_set', default='data/Vaihingen/splits/0.5/lab_files.txt', help='txt file labset')
    parser.add_argument('-unlab_set', default='data/Vaihingen/splits/0.5/unlab_files.txt', help='txt file unlabset')
    parser.add_argument('-per_lab', default=0.5, type=float, help='labeled data of test set (default: 0.1)')
    parser.add_argument('-norm', default=None, help='norm list')

    # MODEL & PARAMS
    parser.add_argument('--model', default='UNet', help='model architecture',
                        choices=['UNet', 'DeepLabv3p']),
    parser.add_argument('--backbone', default='resnet50', help='backbone of model architecture',
                        choices=['resnet50', 'resnet101']),
    parser.add_argument('--seed', type=int, default=2023, help='Random seed for reproducability.')
    parser.add_argument('--patch_size', default=512, type=int, help='size of each patch')
    parser.add_argument('--class_num', default=6, type=int, help='number of classes')
    parser.add_argument('--in_channel', default=3, type=int, help='namely the number of image bands')
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size (default: 8), this is the total')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD (default: 0.9)')
    parser.add_argument('--total_epoch', default=50, type=int, help='total epoch')
    parser.add_argument('--iter_per_epoch', default=200, type=int, help='iteration per epoch')

    # SSL setting
    parser.add_argument('--out_dim', type=int, default=64, help='out dimension of feature (default:64)')
    parser.add_argument('--temperature', default=0.05, type=float, help='softmax temperature (default: 0.5)')
    parser.add_argument('--apply_cc', default=True, help='whether to apply cc loss')
    parser.add_argument('--apply_unsup', default=True, help='whether to apply unsupervised loss')
    parser.add_argument('--unsupervised_mode', default='class_threshold', help='unsupervised loss mode')
    parser.add_argument('--unsup_threshold', default=0.95, type=float)
    parser.add_argument('--strong_threshold', default=0.98, type=float)

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    for dataset_name in ['Vaihingen', 'Potsdam', 'LoveDA-Urban', 'LoveDA-Rural']:
        for i in [1 / 2, 1 / 64, 1 / 4, 1 / 8, 1 / 16, 1 / 32]:

            if dataset_name == 'Vaihingen':
                args.in_channel = 4
                args.class_num = 6
                args.norm = [0, 255, 0, 255, 0, 255, 0, 255]
                args.iter_per_epoch = 200

            elif dataset_name == 'Potsdam':
                args.in_channel = 5
                args.class_num = 6
                args.norm = [3, 255, 1, 255, 1, 255, 0, 255, 0, 255]
                args.iter_per_epoch = 200

            elif dataset_name.__contains__('Urban'):
                args.in_channel = 3
                args.class_num = 7
                args.norm = [0, 255, 0, 255, 0, 255]
                args.iter_per_epoch = 500

            elif dataset_name.__contains__('Rural'):
                args.in_channel = 3
                args.class_num = 7
                args.norm = [0, 255, 0, 255, 0, 255]
                args.iter_per_epoch = 500

            else:
                raise NotImplementedError

            args.dataset_name = dataset_name
            args.train_data = 'data/{}'.format(dataset_name)
            args.lab_set = 'data/{}/splits/{}/lab_files.txt'.format(dataset_name, i)
            args.unlab_set = 'data/{}/splits/{}/unlab_files.txt'.format(dataset_name, i)
            args.train_set = args.lab_set
            args.per_lab = i
            os.makedirs('logging/{}'.format(dataset_name), exist_ok=True)

            # --------------------------------------------------------------------------------

            torch.cuda.empty_cache()
            main_CASSF(args, bviz=False)
