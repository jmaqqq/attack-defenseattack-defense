import torch
import torch.nn.functional as F

def fgsm_attack(model,images,labels,eps=8/256):
    mean=images.new_tensor([0.4919,0.4822,0.4465]).view(1,3,1,1)
    std=images.new_tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
    x=(images.detach()*std+mean).clone()
    x.requires_grad_(True)
    outputs=model((x-mean)/std)
    loss=F.cross_entropy(outputs,labels)
    grad=torch.autograd.grad(loss,x)[0]
    x_adv = torch.clamp(grad.sign() * eps + x, 0, 1)
    x_adv=(x_adv-mean)/std
    return x_adv
