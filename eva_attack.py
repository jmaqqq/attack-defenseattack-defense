import torch
from fgsm import fgsm_attack

def eva_fgsm(model,test_loader,device,eps=8/255):
    model.eval()
    total=0
    clean_correct=0
    adv_correct=0
    attack_success=0

    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)

        with torch.no_grad():
            outputs=model(images)
            clean_pred=outputs.argmax(dim=1)

        adv_images=fgsm_attack(model,images,labels,eps)
        with torch.no_grad():
            adv_outputs=model(adv_images)
            adv_pred=adv_outputs.argmax(dim=1)

        clean_mask=clean_pred.eq(labels)
        adv_mask=adv_pred.eq(labels)
        clean_correct+=clean_mask.sum().item()
        adv_correct+=adv_mask.sum().item()
        # 某张图片在攻击之前就已经分类错误，那么即使攻击后仍然分类错误，也不能说是 FGSM 攻击成功了
        attack_success += (clean_mask & ~adv_mask).sum().item()
        total += labels.size(0)

    clean_acc = clean_correct / total * 100
    adv_acc = adv_correct / total * 100

    asr = (attack_success / clean_correct * 100 if clean_correct > 0 else 0)

    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Attack Success Rate: {asr:.2f}%")
    return clean_acc, adv_acc, asr






