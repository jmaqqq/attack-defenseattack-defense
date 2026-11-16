import torch
import torch.nn.functional as F

def pgd_attack(model,images,labels,eps=8/255,alpha=2/255,steps=10,random_start=True):
    mean=images.new_tensor([0.4919,0.4822,0.4465]).view(1,3,1,1)
    std=images.new_tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
    x=(images.detach()*std+mean).clone()
    if random_start:
        # x + 随机张量会进行加法运算，并创建一个新的结果张量。因此，x_adv 已经拥有独立的存储空间，不需要再使用 clone()。
        x_adv=x+torch.empty_like(x).uniform_(-eps,eps)
        x_adv=torch.clip(x_adv,0,1)
    else:
        x_adv=x.clone()

    for step in range(steps):
        x_adv.requires_grad_(True)
        outputs=model((x_adv-mean)/std)
        loss=F.cross_entropy(outputs,labels)
        grad=torch.autograd.grad(loss,x_adv)[0]
        x_adv=x_adv.detach()+alpha*grad.sign()
        x_adv=torch.clamp(x_adv,(x-eps),(x+eps))
        x_adv=torch.clamp(x_adv,0,1)
    return x_adv






