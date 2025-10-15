from dataset.semi import *
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.segformer import SegFormer
from utils import count_params, color_map, SegmentationMetrics, compute_hausdorff
from PIL import Image
import numpy as np
import time
import argparse
from copy import deepcopy
import os
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['suim', 'caveseg', 'uws'], default='suim')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet101', 'b5'], default='resnet101')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'segformer'],
                        default='deeplabv3plus')
    parser.add_argument('--classes', type=int, default=13)
    parser.add_argument('--conf_thresh', type=float, default=0.95)

    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--reliable-id-path', type=str)

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.reliable_id_path and not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset1(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # 
    print('\n================> Total stage 1/6: Initial training phase')

    trainset_u = SemiDataset2(args.dataset, args.data_root, 'train_u',
                             args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiDataset2(args.dataset, args.data_root, 'train_l',
                             args.crop_size, args.labeled_id_path, nsample=len(trainset_u.ids))

    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size,
                               pin_memory=True, num_workers=1, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size,
                               pin_memory=True, num_workers=1, drop_last=True)
    
    model, optimizer = init_basic_elems(args)
    print(f'\nParams: %.1fM' % count_params(model))

    best_model = train_fix(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, args)

    # 
    print('\n\n\n================> Total stage 2/6: Pseudo labels for all unlabeled images')

    dataset = SemiDataset1(args.dataset, args.data_root, 'label', None, None, unlabeled_id_path=args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    label(best_model, dataloader, args)

    # 
    print('\n\n\n================> Total stage 3/6: Select reliable images')

    trainset = SemiDataset1(args.dataset, args.data_root, 'train', args.crop_size,
                           labeled_id_path=args.labeled_id_path)
    dataloader = DataLoader(trainset, batch_size=1, shuffle=True,
                            pin_memory=True, num_workers=4, drop_last=True)
    select_reliable(best_model, dataloader, args)

    # 
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    trainset = SemiDataset0(args.dataset, args.data_root, 'semi_train', args.crop_size,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
    trainloader_l2 = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    cur_unlabeled_id_path2 = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    trainset_u = SemiDataset2(args.dataset, args.data_root, 'train_u',
                             args.crop_size, cur_unlabeled_id_path2)
    trainloader_u2 = DataLoader(trainset_u, batch_size=args.batch_size,
                               pin_memory=True, num_workers=1, drop_last=True)
    
    model, optimizer = init_basic_elems(args)
    best_model = train_fix(model, trainloader_l2, trainloader_u2, valloader, criterion, optimizer, args)

    # 
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')

    dataset = SemiDataset1(args.dataset, args.data_root, 'label', None, None, unlabeled_id_path=args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    label(best_model, dataloader, args)

    # 
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training')

    trainset = SemiDataset1(args.dataset, args.data_root, 'semi_train', args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)
    train(model, trainloader, valloader, criterion, optimizer, args)


def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'segformer': SegFormer}
    model = model_zoo[args.model](args.backbone, args.classes)

    head_lr_multiple = 10.0
    optimizer = SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
         'lr': args.lr * head_lr_multiple}
    ], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()
    return model, optimizer


def train_fix(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader_u) * args.epochs
    best_metrics = {
        'miou': 0.0,
        'mpa': 0.0,
        'mprecision': 0.0,
        'mhausdorff': float('inf')
    }
    best_model = None

    for epoch in range(args.epochs):
        print(f"\n==> Epoch {epoch+1}/{args.epochs}, learning rate = {optimizer.param_groups[0]['lr']:.6f}")

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        model.train()
        total_loss = 0.0

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix)[0].detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]

            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp, _ = model(torch.cat((img_x, img_u_w)), need_fp=True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_x_fp, pred_u_w_fp = preds_fp.split([num_lb, num_ulb])

            pred_u_s1 = model(img_u_s1)[0]

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]
            
            loss_x0 = criterion(pred_x, mask_x)
            loss_x1 = criterion(pred_x_fp, mask_x)
            loss_x = (loss_x0 + loss_x1) / 2.0

            loss_u_s1 = criterion(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= args.conf_thresh) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_w_fp = criterion(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= args.conf_thresh) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()
            
            loss_u_s = (loss_u_s1 + loss_u_w_fp) / 2.0
            loss = (loss_x + loss_u_s) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iters += 1
            
            lr = 0.001 * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10

        avg_loss = total_loss / len(trainloader_u)
        print(f"Train Loss: {avg_loss:.4f}")

        # validation
        metrics = SegmentationMetrics(args.classes)
        all_preds = []
        all_gts = []
        
        model.eval()
        tbar = tqdm(valloader)
        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)[0]
                pred = torch.argmax(pred, dim=1)
                
                pred_np = pred.cpu().numpy()
                mask_np = mask.numpy()
                
                metrics.add_batch(pred_np, mask_np)
                all_preds.extend(pred_np)
                all_gts.extend(mask_np)
                
                current_results = metrics.evaluate()
                tbar.set_description(f'mIoU: {current_results["miou"]*100:.2f}')

        results = metrics.evaluate()
        mhausdorff, _ = compute_hausdorff(all_preds, all_gts, args.classes, target_size=(args.crop_size, args.crop_size))
        
        print(f"Validation Results - Epoch {epoch+1}:")
        print(f"mIoU: {results['miou']*100:.2f}%, mPA: {results['mpa']*100:.2f}%, mPrecision: {results['mprecision']*100:.2f}%, mHausdorff: {mhausdorff:.4f}")

        if results['miou'] > best_metrics['miou']:
            best_metrics = {
                'miou': results['miou'],
                'mpa': results['mpa'],
                'mprecision': results['mprecision'],
                'mhausdorff': mhausdorff
            }
            best_model = deepcopy(model)
            
            # save model
            if os.path.exists(os.path.join(args.save_path, f"{args.model}_best.pth")):
                os.remove(os.path.join(args.save_path, f"{args.model}_best.pth"))
            torch.save(model.module.state_dict(), os.path.join(args.save_path, f"{args.model}_best.pth"))
            print(f"Saved best model (mIoU: {best_metrics['miou']*100:.2f}%)")

    print("\nStage Training Complete! Best Metrics:")
    print(f"mIoU: {best_metrics['miou']*100:.2f}%, mPA: {best_metrics['mpa']*100:.2f}%, mPrecision: {best_metrics['mprecision']*100:.2f}%, mHausdorff: {best_metrics['mhausdorff']:.4f}")

    return best_model


def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs
    best_metrics = {
        'miou': 0.0,
        'mpa': 0.0,
        'mprecision': 0.0,
        'mhausdorff': float('inf')
    }

    for epoch in range(args.epochs):
        print(f"\n==> Epoch {epoch+1}/{args.epochs}, learning rate = {optimizer.param_groups[0]['lr']:.6f}")


        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask, _) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)[0]
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iters += 1
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            tbar.set_description(f'Loss: {total_loss/(i+1):.3f}')

        avg_loss = total_loss / len(trainloader)
        print(f"Train Loss: {avg_loss:.4f}")

        metrics = SegmentationMetrics(args.classes)
        all_preds = []
        all_gts = []
        
        model.eval()
        tbar = tqdm(valloader)
        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)[0]
                pred = torch.argmax(pred, dim=1)
                
                pred_np = pred.cpu().numpy()
                mask_np = mask.numpy()
                
                metrics.add_batch(pred_np, mask_np)
                all_preds.extend(pred_np)
                all_gts.extend(mask_np)
                
                current_results = metrics.evaluate()
                tbar.set_description(f'mIoU: {current_results["miou"]*100:.2f}')

        results = metrics.evaluate()
        mhausdorff, _ = compute_hausdorff(all_preds, all_gts, args.classes, target_size=(args.crop_size, args.crop_size))

        print(f"Validation Results - Epoch {epoch+1}:")
        print(f"mIoU: {results['miou']*100:.2f}%, mPA: {results['mpa']*100:.2f}%, mPrecision: {results['mprecision']*100:.2f}%, mHausdorff: {mhausdorff:.4f}")
        
        if results['miou'] > best_metrics['miou']:
            best_metrics = {
                'miou': results['miou'],
                'mpa': results['mpa'],
                'mprecision': results['mprecision'],
                'mhausdorff': mhausdorff
            }

            prev_best_path = os.path.join(args.save_path, f"{args.model}_{args.backbone}_best.pth")
            if os.path.exists(prev_best_path):
                os.remove(prev_best_path)
            torch.save(model.module.state_dict(), prev_best_path)
            print(f"Saved best model (mIoU: {best_metrics['miou']*100:.2f}%)")

    print("\nFinal Training Complete! Best Metrics:")
    print(f"mIoU: {best_metrics['miou']*100:.2f}%, mPA: {best_metrics['mpa']*100:.2f}%, mPrecision: {best_metrics['mprecision']*100:.2f}%, mHausdorff: {best_metrics['mhausdorff']:.4f}")
    
    return model


def select_reliable(model, dataloader, args):
    model.eval()
    vecs_all = torch.zeros((args.classes, 256)).cuda()
    num = torch.zeros((args.classes)).cuda()
    
    tbar = tqdm(dataloader)
    with torch.no_grad():
        for img, mask, _ in tbar:
            img = img.cuda()
            mask = mask.cuda()
            _, feat = model(img)
            
            for index in range(args.classes):
                mask_cp = mask.clone()
                mask_cp[mask == index] = 1
                mask_cp[mask != index] = 0
                vec = Weighted_GAP(feat, mask_cp).view(-1)
                vecs_all[index, :] += vec
                num[index] += 1

    num_2d = num.reshape(len(num), 1)
    vecs_all = vecs_all / num_2d

    unlabeled_dataset = SemiDataset1(args.dataset, args.data_root, 'select_unlabeled', args.crop_size,
                           unlabeled_id_path=args.unlabeled_id_path, pseudo_mask_path=args.pseudo_mask_path)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=True,
                            pin_memory=True, num_workers=4, drop_last=True)
    tbar = tqdm(unlabeled_dataloader)
    id_to_reliability = []
    
    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            mask = mask.cuda()
            mask_np = mask.clone().squeeze(0).cpu().numpy()
            _, feat = model(img)
            n, c, _, _ = feat.size()

            cls_set = list(np.unique(mask_np))
            reliability = 0
            if len(cls_set) == 0:
                reliability = 1e16
            else:
                for cls in cls_set:
                    anchor = vecs_all[cls]
                    mask_cp = mask.clone()
                    mask_cp[mask == cls] = 1
                    mask_cp[mask != cls] = 0

                    simi = F.cosine_similarity(feat, anchor.view(n, c, 1, 1))
                    simi = F.interpolate(simi.unsqueeze(1), size=mask.size()[-2:], mode='bilinear').view(mask.size())
                    bce_loss = F.binary_cross_entropy(simi, mask_cp.float(), reduction='none')
                    loss = bce_loss[mask != 255].mean()
                    reliability += loss

            tbar.set_description(f'reliability: {reliability:.3f}')
            id_to_reliability.append((id[0], reliability))
    
    id_to_reliability.sort(key=lambda elem: elem[1], reverse=False)

    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')

    print(f"Reliable selection threshold: {id_to_reliability[len(id_to_reliability)//2][1].cpu().numpy()}")

def Weighted_GAP(supp_feat, mask):
    if len(mask.size()) != 4:
        mask = mask.unsqueeze(1).float()
    if supp_feat.size() != mask.size():
        supp_feat = F.interpolate(supp_feat, size=mask.size()[-2:], mode='bilinear')

    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metrics = SegmentationMetrics(args.classes)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img)[0]
            pred = torch.argmax(pred, dim=1).cpu()

            metrics.add_batch(pred.numpy(), mask.numpy())
            current_results = metrics.evaluate()

            pred_img = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred_img.putpalette(cmap)
            pred_path = os.path.join(args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1]))
            pred_img.save(pred_path)

            tbar.set_description(f'mIoU: {current_results["miou"]*100:.2f}')

    final_results = metrics.evaluate()
    print(f"Pseudo labeling complete. mIoU: {final_results['miou']*100:.2f}%")

if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'caveseg': 40, 'suim': 40, 'uws': 40}[args.dataset]
    if args.lr is None:
        args.lr = {'caveseg': 0.001, 'suim': 0.001, 'uws': 0.001}[args.dataset]
    if args.crop_size is None:
        args.crop_size = {'caveseg': 512, 'suim': 512, 'uws': 512}[args.dataset]

    print(args)
    main(args)
