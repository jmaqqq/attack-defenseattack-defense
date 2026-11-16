import torch
import torch.nn.functional as F

def mi_fgsm_attack(model,images,labels,eps=8/255,alpha=2/255,steps=10,decay=1.0):
    mean=images.new_tensor([0.4919,0.4822,0.4465]).view(1,3,1,1)
    std=images.new_tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)
    x=(images.detach()*std+mean).clone()
    x_adv=x.clone()
    vel=torch.zeros_like(x_adv)

    for step in range(steps):
        x_adv=x_adv.detach().requires_grad_(True)
        outputs=model((x_adv-mean)/std)
        loss=F.cross_entropy(outputs,labels)
        grad = torch.autograd.grad(loss, x_adv)[0]
        vel=decay*vel+grad/(grad.abs().mean(dim=(1,2,3),keepdim=True).clamp_min(1e-12))
        x_adv = x_adv.detach() + alpha * vel.sign()
        x_adv = torch.clamp(x_adv, min=x - eps, max=x + eps)
        x_adv = torch.clamp(x_adv, 0, 1)

    return (x_adv-mean)/std