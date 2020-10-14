import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from KD.base_kd import hinton_distillation, hinton_distillation_wo_ce
import KD.od_distiller as od_distiller
import os
import DA.DA_datasets as DA_datasets
import cmodels.ResNet as ResNet
import cmodels.DAN_model as DAN_model
from utils import eval, adjust_learning_rate


def mmd_hinton_train_alt(current_epoch, epochs, teacher_model, student_model, optimizer_da, optimizer_kd, device,
                         source_dataloader, target_dataloader, T, alpha, beta, kd_loss_fn, is_debug=False,  **kwargs):


    teacher_model.train()
    student_model.train()

    total_loss = 0.
    teacher_da_temp_loss = 0.
    kd_temp_loss = 0.
    kd_target_loss = 0.
    kd_source_loss = 0.

    iter_source = iter(source_dataloader)
    iter_target = iter(target_dataloader)

    for i in range(1, len(source_dataloader) + 1):

        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()

        if data_source.shape[0] != data_target.shape[0]:
            if data_target.shape[0] < source_dataloader.batch_size:
                iter_target = iter(target_dataloader)
                data_target, _ = iter_target.next()

            if data_source.shape[0] < source_dataloader.batch_size:
                data_target = data_target[:data_source.shape[0]]


        data_source, label_source = data_source.to(device), label_source.to(device)
        data_target = data_target.to(device)

        # Teacher domain adaptation
        optimizer_da.zero_grad()
        teacher_label_source_pred, teacher_loss_mmd, _ = teacher_model(data_source, data_target)
        teacher_source_loss_cls = F.nll_loss(F.log_softmax(teacher_label_source_pred, dim=1), label_source)
        gamma = 2 / (1 + np.exp(-10 * (i) / len(source_dataloader))) - 1
        teacher_da_mmd_loss = (1 - beta ) * (teacher_source_loss_cls + gamma * teacher_loss_mmd)
        teacher_da_temp_loss += teacher_da_mmd_loss.mean().item()

        # Possible to do end2end or alternative here: For now it's alternative
        teacher_da_mmd_loss.mean().backward()
        optimizer_da.step() # May need to have 2 optimizers
        optimizer_da.zero_grad()

        #Knowledge distillation: We only learn on target logits now

        optimizer_kd.zero_grad()
        teacher_source_logits, teacher_loss_mmd, teacher_target_logits = teacher_model(data_source, data_target)
        student_source_logits, student_loss_mmd, student_target_logits = student_model(data_source, data_target)

        source_kd_loss = hinton_distillation(teacher_source_logits, student_source_logits, label_source, T, alpha, kd_loss_fn).abs()
        target_kd_loss = hinton_distillation_wo_ce(teacher_target_logits, student_target_logits, T, kd_loss_fn).abs()

        kd_source_loss += source_kd_loss.mean().item()
        kd_target_loss += target_kd_loss.mean().item()

        kd_loss = beta * (target_kd_loss + source_kd_loss)
        kd_temp_loss += kd_loss.mean().item()
        total_loss += teacher_da_mmd_loss.mean().item() + kd_loss.mean().item()

        kd_loss.mean().backward()
        optimizer_kd.step()
        optimizer_kd.zero_grad()

        if is_debug:
            break

    del kd_loss
    del teacher_da_mmd_loss
    # torch.cuda.empty_cache()
    return total_loss / len(source_dataloader), teacher_da_temp_loss / len(source_dataloader), \
           kd_temp_loss / len(source_dataloader), kd_source_loss / len(source_dataloader), kd_target_loss / len(source_dataloader)


def mmd_hinton_alt(init_lr_da, init_lr_kd, device, epochs, T, alpha, growth_rate, init_beta,
                   source_dloader,target_dloader, target_testloader, optimizer_da, optimizer_kd, teacher_model, student_model,
                   is_scheduler_da=True, is_scheduler_kd=False, scheduler_da=None, scheduler_kd=None, is_debug=False, kd_loss_fn=F.kl_div, **kwargs):


    best_student_acc = 0.
    best_teacher_acc = 0.
    epochs += 1

    for epoch in range(1, epochs):

        beta = init_beta * torch.exp(growth_rate * (epoch - 1))
        beta = beta.to(device)
        if is_scheduler_da:
            new_lr_da = init_lr_da / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            adjust_learning_rate(optimizer_da, new_lr_da)

        if is_scheduler_kd:
            new_lr_kd = init_lr_kd / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            adjust_learning_rate(optimizer_kd, new_lr_kd)

        total_loss, da_loss, kd_loss, kd_source_loss, kd_target_loss = mmd_hinton_train_alt(epoch, epochs, teacher_model, student_model, optimizer_da,
                                                            optimizer_kd, device, source_dloader, target_dloader, T,
                                                            alpha, beta, kd_loss_fn, is_debug)

        teacher_target_acc = eval(teacher_model, device, target_testloader, is_debug)
        student_target_acc = eval(student_model, device, target_testloader, is_debug)
        print(f'epoch : {epoch}, acc : {student_target_acc}, teacher acc : {teacher_target_acc}')
        if student_target_acc > best_student_acc:
            best_student_acc = student_target_acc
            torch.save({'student_model': student_model.state_dict(), 'acc': best_student_acc, 'epoch': epoch},
                       "./kd_da_alt_pth_student_best_model.pth")

        if scheduler_da is not None:
            scheduler_da.step()

        if scheduler_kd is not None:
            scheduler_kd.step()

    return best_student_acc

def main():


    batch_size = 16
    test_batch_size = 32

    webcam = os.path.expanduser("./datasets/webcam/images")
    amazon = os.path.expanduser("./datasets/amazon/images")
    dslr = os.path.expanduser("./datasets/dslr/images")
    is_debug = False

    epochs = 400
    init_lr_da = 0.001
    init_lr_kd = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    device = torch.device("cuda")
    T = 20
    alpha = 0.3
    init_beta = 0.1
    end_beta = 0.9

    student_pretrained = True

    if torch.cuda.device_count() > 1:
        teacher_model = nn.DataParallel(DAN_model.DANNet_ResNet(ResNet.resnet101, True)).to(device)
        student_model = nn.DataParallel(DAN_model.DANNet_ResNet(ResNet.resnet34, student_pretrained)).to(device)
    else:
        teacher_model = DAN_model.DANNet_ResNet(ResNet.resnet101, True).to(device)
        student_model = DAN_model.DANNet_ResNet(ResNet.resnet34, student_pretrained).to(device)

    growth_rate = torch.log(torch.FloatTensor([end_beta / init_beta])) / torch.FloatTensor([epochs])


    optimizer_da = torch.optim.SGD(list(teacher_model.parameters()) + list(student_model.parameters()), init_lr_da,
                                momentum=momentum, weight_decay=weight_decay)

    optimizer_kd = torch.optim.SGD(list(teacher_model.parameters()) + list(student_model.parameters()), init_lr_kd,
                                momentum=momentum, weight_decay=weight_decay)


    source_dataloader, target_dataloader, target_testloader = DA_datasets.get_source_target_loader("Office31",
                                                                                                   amazon,
                                                                                                   webcam,
                                                                                                   batch_size, 0)

    mmd_hinton_alt(init_lr_da, init_lr_kd, device, epochs, T, alpha, growth_rate, init_beta, source_dataloader,
               target_dataloader, target_testloader, optimizer_da, optimizer_kd, teacher_model, student_model, is_scheduler_kd=True, is_scheduler_da=True, is_debug=False)


if __name__ == "__main__":
    main()
