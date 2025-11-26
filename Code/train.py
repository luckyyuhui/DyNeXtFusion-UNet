import os
import time
import datetime
import torch
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import VOCSegmentation
import transforms as T
from src import DyNeXtFusionUNet


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.388230, 0.392696, 0.361650), std=(0.233420, 0.217205, 0.218831)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.388375,0.392255,0.361999), std=(0.232980,0.217078,0.218584)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.388230, 0.392696, 0.361650), std=(0.233420, 0.217205, 0.218831)):
    base_size = 512
    crop_size = 512

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    model = DyNeXtFusionUNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    mean=(0.388230, 0.392696, 0.361650)
    std=(0.233420, 0.217205, 0.218831)

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # VOCdevkit -> VOC -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                 transforms=get_transform(train=True, mean=mean, std=std),
                                    txt_name='train.txt')

    # VOCdevkit -> VOC -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                               transforms=get_transform(train=False, mean=mean, std=std),
                                  txt_name='val.txt')

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)


    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None


    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

        ckpt_state = checkpoint.get('model', None)
        if ckpt_state is None:
            if isinstance(checkpoint, dict) and any(k.startswith('module.') or k in ['state_dict', 'model_state'] for k in checkpoint.keys()):
                # find likely state_dict
                if 'state_dict' in checkpoint:
                    ckpt_state = checkpoint['state_dict']
                elif 'model_state' in checkpoint:
                    ckpt_state = checkpoint['model_state']
                else:
                    ckpt_state = checkpoint
            else:
                ckpt_state = checkpoint

        def _strip_module_prefix(state_dict):
            new_state = {}
            for k, v in state_dict.items():
                new_key = k[7:] if k.startswith('module.') else k
                new_state[new_key] = v
            return new_state

        model_state_keys = list(model.state_dict().keys())
        ckpt_keys = list(ckpt_state.keys()) if isinstance(ckpt_state, dict) else []

        if isinstance(ckpt_state, dict):
            if any(k.startswith('module.') for k in ckpt_keys) and not any(k.startswith('module.') for k in model_state_keys):
                ckpt_state = _strip_module_prefix(ckpt_state)
            elif not any(k.startswith('module.') for k in ckpt_keys) and any(k.startswith('module.') for k in model_state_keys):
                ckpt_state = {'module.' + k: v for k, v in ckpt_state.items()}

        try:
            model.load_state_dict(ckpt_state)
        except Exception:
            model.load_state_dict(ckpt_state, strict=False)
            print("Warning: loaded checkpoint with strict=False (some keys mismatch)")

        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except Exception as e:
                print(f"Warning: failed to load optimizer state: {e}")

        if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except Exception as e:
                print(f"Warning: failed to load lr_scheduler state: {e}")

        # AMP scaler
        if args.amp and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler'])
            except Exception as e:
                print(f"Warning: failed to load AMP scaler state: {e}")

        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        else:
            args.start_epoch = args.start_epoch

        print(f"Resumed from {args.resume}, start_epoch={args.start_epoch}")

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch model training")

    parser.add_argument("--data-path", default="", help="root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)

    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # 训练过程配置
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default="", help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
