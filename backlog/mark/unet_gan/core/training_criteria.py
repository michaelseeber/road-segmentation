def soft_dice_loss(pred, true, smooth=1):
    true = true.flatten()
    pred = pred.flatten()
    intersection = (true * pred).sum()
    return 1 - (2. * intersection + smooth) / ((true.sum() + pred.sum()) + smooth)
    
def accuracy(pred, true):
    _pred = pred.detach()
    _true = true.detach()
    _pred = _pred >= 0.5
    _true = _true >= 0.5
    return _pred.eq(_true).sum().float() / _pred.numel()

def l1_loss_sum(pred, true):
    return (pred - true).abs().sum()

def l1_loss_mean(pred, true):
    return (pred - true).abs().mean()

def l2_loss_sum(pred, true):
    return ((pred - true)**2).sum()

def l2_loss_mean(pred, true):
    return ((pred - true)**2).mean()