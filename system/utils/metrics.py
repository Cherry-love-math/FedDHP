import torch

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    pred = logits.topk(maxk, dim=1, largest=True, sorted=True)[1].t()  # [K,B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    B = target.size(0)
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / B).item())
    return res  # e.g. [top1, top5]
