import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target

def ce_dice_criterion(inputs, target, loss_weight=None,
                      num_classes: int = 2, dice_weight: float = 1.0,
                      aux_weight: float = 0.5, ignore_index: int = 255):

    if not isinstance(inputs, dict):
        inputs = {'out': inputs}

    if loss_weight is not None and not torch.is_tensor(loss_weight):
        loss_weight = torch.as_tensor(loss_weight, dtype=torch.float32)
    if loss_weight is not None:
        loss_weight = loss_weight.to(target.device).float()

    target = target.long()

    dice_target = build_target(target, num_classes, ignore_index)

    losses = {}
    for name, logits in inputs.items():
        logits = logits.float()
        # CrossEntropy
        ce = nn.functional.cross_entropy(logits, target, weight=loss_weight, ignore_index=ignore_index)
        # Dice
        d = dice_loss(logits, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = ce + dice_weight * d

    if len(losses) == 1:
        return next(iter(losses.values()))

    main = losses.get('out', None)
    aux = losses.get('aux', None)
    if main is None:
        first_key = next(iter(losses.keys()))
        main = losses[first_key]
        aux = None
        for k in losses.keys():
            if k != first_key:
                aux = losses[k]
                break

    if aux is None:
        return main
    else:
        return main + aux_weight * aux

def evaluate(model, data_loader, device, num_classes, ignore_index=255):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice_metric = utils.DiceCoefficient(num_classes=num_classes, ignore_index=ignore_index)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            if isinstance(output, dict):
                out = output.get('out', next(iter(output.values())))
            else:
                out = output
            preds = out.argmax(dim=1)
            confmat.update(target.flatten(), preds.flatten())
            dice_metric.update(out, target)

        confmat.reduce_from_all_processes()
        dice_metric.reduce_from_all_processes()

    dice_val = dice_metric.value
    if isinstance(dice_val, torch.Tensor):
        try:
            dice_out = dice_val.mean().item()
        except Exception:
            dice_out = float(dice_val)
    else:
        dice_out = float(dice_val)
    return confmat, dice_out

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler=None, print_freq=10, scaler=None,
                    dice_weight: float = 1.0, aux_weight: float = 0.5):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], dtype=torch.float32, device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        target = target.to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(image)
            loss = ce_dice_criterion(outputs, target,
                                     loss_weight=loss_weight,
                                     num_classes=num_classes,
                                     dice_weight=dice_weight,
                                     aux_weight=aux_weight,
                                     ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
