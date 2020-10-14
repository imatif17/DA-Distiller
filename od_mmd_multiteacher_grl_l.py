import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from KD.base_kd import hinton_distillation, hinton_distillation_sw, hinton_distillation_wo_ce
#import KD.od_distiller as od_distiller
import os
import DA.DA_datasets2 as DA_datasets
import cmodels.ResNet as ResNet
#import cmodels.DAN_model as DAN_model
import cmodels.DANN_GRL as DANN_GRL
from utils import eval, adjust_learning_rate, get_sub_dataset_name

def grl_multi_target_hinton_train_alt(current_ep, epochs, teacher_models, student_model, optimizer_das, optimizer_kd, device,
                         source_dataloader, targets_dataloader, T, alpha, beta, gamma, batch_norm, is_cst, is_debug=False,  **kwargs):

    if batch_norm:
        for teacher_model in teacher_models:
            teacher_model.train()
        student_model.train()

    total_losses = torch.zeros(3)
    teacher_da_temp_losses = torch.zeros(3)
    kd_temp_losses = torch.zeros(3)
    kd_target_loss = 0.
    kd_source_loss = 0.

    iter_targets = [0] * len(targets_dataloader)
    for i, d in enumerate(targets_dataloader):
        iter_targets[i] = iter(d)

    iter_source = iter(source_dataloader)

    for i in range(1, len(source_dataloader) + 1):

        data_source, label_source = iter_source.next()
        data_source = data_source.to(device)
        label_source = label_source.to(device)

        for ix, it in enumerate(iter_targets):
            try:
                data_target, _ = it.next()
            except StopIteration:
                it = iter(targets_dataloader[ix])
                data_target, _ = it.next()

            if data_target.shape[0] != data_source.shape[0]:
                data_target = data_target[: data_source.shape[0]]
            data_target = data_target.to(device)
            optimizer_das[ix].zero_grad()
            p = float(i + (current_ep -1) * len(source_dataloader)) / epochs / len(source_dataloader)
            delta = 2. / (1. + np.exp(-10 * p)) - 1
            teacher_label_source_pred, teacher_source_loss_adv = teacher_models[ix](data_source, delta)
            teacher_source_loss_cls = F.cross_entropy(F.log_softmax(teacher_label_source_pred, dim=1), label_source)

            _, teacher_target_loss_adv = teacher_models[ix](data_target, delta, source=False)
            teacher_loss_adv = teacher_source_loss_adv + teacher_target_loss_adv

            teacher_da_grl_loss = (1 - beta) * (teacher_source_loss_cls + gamma * teacher_loss_adv)
            teacher_da_temp_losses[ix] += teacher_da_grl_loss.mean().item()

            teacher_da_grl_loss.mean().backward()
            optimizer_das[ix].step()
            optimizer_das[ix].zero_grad()


            optimizer_kd.zero_grad()
            teacher_source_logits, _  = teacher_models[ix](data_source, delta, source=True)
            teacher_target_logits, _ = teacher_models[ix](data_target, delta, source=True)

            student_source_logits, _  = student_model(data_source, delta, source=True)
            student_target_logits, student_target_loss_adv = student_model(data_target, delta, source=False)

            source_kd_loss = hinton_distillation_sw(teacher_source_logits, student_source_logits, label_source, T, alpha).abs()
            if is_cst:
                target_kd_loss = hinton_distillation_wo_ce(teacher_target_logits, student_target_logits, T).abs() + alpha * student_target_loss_adv

            kd_source_loss += source_kd_loss.mean().item()
            kd_target_loss += target_kd_loss.mean().item()

            kd_loss = beta * (target_kd_loss + source_kd_loss)
            kd_temp_losses[ix] += kd_loss.mean().item()
            total_losses[ix] += teacher_da_grl_loss.mean().item() + kd_loss.mean().item()

            kd_loss.mean().backward()
            optimizer_kd.step()
            optimizer_kd.zero_grad()

        if is_debug:
            break

    del kd_loss
    del teacher_da_grl_loss
    return total_losses / len(source_dataloader), teacher_da_temp_losses / len(source_dataloader), kd_temp_losses / len(source_dataloader)


def grl_multi_target_hinton_alt(init_lr_da, init_lr_kd, device, epochs, T, alpha, gamma, growth_rate, init_beta,
                   source_dloader, targets_dloader, targets_testloader, optimizer_das, optimizer_kd, teacher_models, student_model,
                   is_scheduler_da=True, is_scheduler_kd=True, scheduler_da=None, scheduler_kd=None, is_debug=False, save_name="", batch_norm=False, is_cst=True, **kwargs):


    best_student_acc = 0.
    best_teacher_acc = 0.
    epochs += 1
    for epoch in range(1, epochs):

        beta = init_beta * torch.exp(growth_rate * (epoch - 1))
        beta = beta.to(device)
        if is_scheduler_da:
            new_lr_da = init_lr_da / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            for optimizer_da in optimizer_das:
                adjust_learning_rate(optimizer_da, new_lr_da)

        if is_scheduler_kd:
            new_lr_kd = init_lr_kd / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
            adjust_learning_rate(optimizer_kd, new_lr_kd)

        total_loss_s1, da_loss_s1, kd_loss_1 = grl_multi_target_hinton_train_alt(epoch, epochs, teacher_models,           
                                                                                 student_model, optimizer_das,
                                                                                 optimizer_kd, device, source_dloader, 
                                                                                 targets_dloader, T,
                                                                                 alpha, beta, gamma, batch_norm, is_cst, 
                                                                                 is_debug, logger=None)

        teachers_targets_acc = np.zeros(len(teacher_models))
        students_targets_acc = np.zeros(len(teacher_models))

        for i, d in enumerate(targets_testloader):
            #teachers_targets_acc[i] = eval(teacher_models[i], device, d, is_debug)
            students_targets_acc[i] = eval(student_model, device, d, is_debug)

        total_target_acc = students_targets_acc.mean()
        print(f'epoch : {epoch}\tacc : {total_target_acc}')

        if total_target_acc > best_student_acc:
            best_student_acc = total_target_acc
            torch.save({'student_model': student_model.state_dict(), 'acc': best_student_acc, 'epoch': epoch},
                       "student_model.pth")


        if scheduler_da is not None:
            scheduler_da.step()

        if scheduler_kd is not None:
            scheduler_kd.step()

    return best_student_acc

def main():

    a = os.path.expanduser('../../datasets/Art')
    c = os.path.expanduser('../../datasets/Clipart')
    r = os.path.expanduser('../../datasets/RealWorld')
    p = os.path.expanduser('../../datasets/Product')
    dataset_name = 'Office31'

    source_dataset_path = a
    target_dataset_path_1 = c
    target_dataset_path_2 = p
    target_dataset_path_3 = r

    init_beta = 0.1
    end_beta = 0.6
    init_lr_da = 0.001
    init_lr_kd = 0.01
    momentum = 0.9
    T = 20
    batch_size = 32
    alpha = 0.5
    gamma = 0.5
    epochs = 400
    scheduler_kd_fn = None
    batch_norm = True
    device = 'cuda'
    weight_decay = 5e-4
    is_scheduler_da = True
    is_scheduler_kd = True

    source_dataloader, targets_dataloader, targets_testloader = DA_datasets.get_source_m_target_loader(dataset_name,
                                                                                                   source_dataset_path,
                                                                                                   [target_dataset_path_1, target_dataset_path_2, target_dataset_path_3],
                                                                                                   batch_size, 0, drop_last=True)

    teacher_model_1 = DANN_GRL.DANN_GRL_Resnet(ResNet.resnet152, True, source_dataloader.dataset.num_classes).to(device)
    teacher_model_2 = DANN_GRL.DANN_GRL_Resnet(ResNet.resnet152, True, source_dataloader.dataset.num_classes).to(device)
    teacher_model_3 = DANN_GRL.DANN_GRL_Resnet(ResNet.resnet152, True, source_dataloader.dataset.num_classes).to(device)
    student_model = DANN_GRL.DANN_GRL_Resnet(ResNet.resnet50, True, source_dataloader.dataset.num_classes).to(device)

    if torch.cuda.device_count() > 1:
        teacher_model_1 = nn.DataParallel(teacher_model_1).to(device)
        teacher_model_2 = nn.DataParallel(teacher_model_2).to(device)
        teacher_model_3 = nn.DataParallel(teacher_model_3).to(device)
        student_model = nn.DataParallel(student_model).to(device)

    teacher_models = [teacher_model_1, teacher_model_2, teacher_model_3]

    growth_rate = torch.zeros(1)
    if init_beta != 0.0:
        growth_rate = torch.log(torch.FloatTensor([end_beta / init_beta])) / torch.FloatTensor([epochs])

    optimizer_da_1 = torch.optim.SGD(teacher_model_1.parameters(), init_lr_da,
                                momentum=momentum, weight_decay=weight_decay)
    optimizer_da_2 = torch.optim.SGD(teacher_model_2.parameters(), init_lr_da,
                                momentum=momentum, weight_decay=weight_decay)
    optimizer_da_3 = torch.optim.SGD(teacher_model_3.parameters(), init_lr_da,
                                momentum=momentum, weight_decay=weight_decay)

    optimizer_das = [optimizer_da_1, optimizer_da_2, optimizer_da_3]

    optimizer_kd = torch.optim.SGD(student_model.parameters(), init_lr_kd,
                                momentum=momentum, weight_decay=weight_decay)

    scheduler_kd = None
    if scheduler_kd_fn is not None:
        scheduler_kd = scheduler_kd_fn(optimizer_kd, scheduler_kd_steps, scheduler_kd_gamma)


    best_student_acc = grl_multi_target_hinton_alt(init_lr_da, init_lr_kd, device, epochs, T, alpha, gamma, growth_rate, init_beta,
                                                source_dataloader, targets_dataloader, targets_testloader, optimizer_das, optimizer_kd,
                                                teacher_models, student_model,
                                                logger=None,
                                                is_scheduler_da=is_scheduler_da, is_scheduler_kd=is_scheduler_kd, scheduler_kd=None, scheduler_da=None,
                                                is_debug = False, batch_norm=batch_norm)

if __name__ == "__main__":
    main()
