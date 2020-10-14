import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import distiller
import os
import DA.DA_datasets as DA_datasets
import cmodels.ResNet as ResNet
import cmodels.DAN_model as DAN_model
import cmodels.DANN_GRL as DANN_GRL
from utils import eval, adjust_learning_rate


def mmd_hinton_train_alt(current_epoch, epochs, distil, criterion, optimizer_da, optimizer_kd, device,
                         source_dataloader, target_dataloader, target_testloader, alpha, gamma, beta, kd_loss_fn, best_teacher_acc, is_debug=False, **kwargs):


    #teacher_model.train()
    #student_model.train()

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
        p = float(i + (current_epoch -1) * len(source_dataloader)) / epochs / len(source_dataloader)
        delta = 2. / (1. + np.exp(-10 * p)) - 1
        teacher_label_source_pred, teacher_source_loss_adv = distil.t_net(data_source, delta)
        teacher_source_loss_cls = F.cross_entropy(F.log_softmax(teacher_label_source_pred, dim=1), label_source)

        _, teacher_target_loss_adv = distil.t_net(data_target, delta, source=False)
        teacher_loss_adv = teacher_source_loss_adv + teacher_target_loss_adv

        teacher_da_grl_loss = (1 - beta) * (teacher_source_loss_cls + gamma * teacher_loss_adv)
        teacher_da_temp_loss += teacher_da_grl_loss.mean().item()

        teacher_da_grl_loss.mean().backward()
        optimizer_da.step()
        optimizer_da.zero_grad()


        teacher_target_acc = eval(distil.t_net, device, target_testloader, is_debug)
        if (teacher_target_acc > best_teacher_acc):
            best_teacher_acc = teacher_target_acc
            torch.save({'teacher_model': distil.t_net.state_dict(), 'acc': best_teacher_acc},
                       "./teacher_model.pth")
        else:
            distil.t_net.load_state_dict(torch.load('./teacher_model.pth')['teacher_model'])

        #Knowledge distillation: We only learn on target logits now

        optimizer_kd.zero_grad()

        _, distil_loss_source = distil(data_source)

        if torch.cuda.device_count() > 1:
            source_outputs_distillar = distil.s_net.module.nforward(data_source)
        else:
            source_outputs_distillar = distil.s_net.nforward(data_source)
        source_outputs_distillar = F.log_softmax(source_outputs_distillar, dim=1)

        _, distil_loss_target = distil(data_target)
        _, student_target_loss_adv = distil.s_net(data_target, delta, source=False)

        class_loss = criterion(source_outputs_distillar, label_source)

        kd_loss = beta * (class_loss + distil_loss_target.sum() / data_source.shape[0] / 10000 + distil_loss_source.sum() / data_target.shape[0] / 10000 + alpha * student_target_loss_adv)
        kd_temp_loss += kd_loss.item()
        total_loss += teacher_grl_mmd_loss.mean().item() + kd_loss.mean().item()

        kd_loss.backward()
        optimizer_kd.step()
        optimizer_kd.zero_grad()

        if is_debug:
            break

    del kd_loss
    del teacher_grl_mmd_loss

    return total_loss / len(source_dataloader), teacher_da_temp_loss / len(source_dataloader), \
           kd_temp_loss / len(source_dataloader), kd_source_loss / len(source_dataloader), kd_target_loss / len(source_dataloader), best_teacher_acc

def mmd_hinton_alt(init_lr_da, init_lr_kd, device, epochs, alpha, gamma, growth_rate, init_beta,
                   source_dloader,target_dloader, target_testloader, optimizer_da, optimizer_kd, distil, criterion,
                   is_scheduler_da=True, is_scheduler_kd=False, scheduler_da=None, scheduler_kd=None, is_debug=False, kd_loss_fn=F.kl_div, best_teacher_acc = 0, **kwargs):


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

        total_loss, da_loss, kd_loss, kd_source_loss, kd_target_loss, best_teacher_acc = mmd_hinton_train_alt(epoch, epochs, distil, criterion, optimizer_da,
                                                            optimizer_kd, device, source_dloader, target_dloader, target_testloader,
                                                            alpha, gamma, beta, kd_loss_fn, best_teacher_acc, is_debug)

        student_target_acc = eval(distil.s_net, device, target_testloader, is_debug)
        print(f'epoch : {epoch}, acc : {student_target_acc}, teacher acc : {best_teacher_acc}')
        if student_target_acc > best_student_acc:
            best_student_acc = student_target_acc
            torch.save({'student_model': distil.s_net.state_dict(), 'acc': best_student_acc, 'epoch': epoch},
                       "./student_model.pth")

        if(epoch == 100 and epoch == 200):
            for param_group in optimizer_kd.param_groups:
                param_group['lr'] = param_group['lr'] * .1

    return best_student_acc

def main():


    batch_size = 16
    test_batch_size = 32

    p = os.path.expanduser("./image-clef/p")
    c = os.path.expanduser("./image-clef/c")
    i = os.path.expanduser("./image-clef/i")
    is_debug = False

    epochs = 400
    init_lr_da = 0.0001
    init_lr_kd = 0.0005
    momentum = 0.9
    weight_decay = 5e-4
    device = torch.device("cuda")
    alpha = 0.5
    gamma = 0.5
    init_beta = 0.1
    end_beta = 0.9

    student_pretrained = True

    source_dataloader, target_dataloader, target_testloader = DA_datasets.get_source_target_loader("ImageClef",
                                                                                                   p,
                                                                                                   c,
                                                                                                   batch_size, 0)
    if torch.cuda.device_count() > 1:
        teacher_model = nn.DataParallel(DANN_GRL.DANN_GRL_Resnet(ResNet.resnet101, True, source_dataloader.dataset.num_classes)).to(device)
        student_model = nn.DataParallel(DANN_GRL.DANN_GRL_Resnet(ResNet.resnet34, student_pretrained, source_dataloader.dataset.num_classes)).to(device)
    else:
        teacher_model = DANN_GRL.DANN_GRL_Resnet(ResNet.resnet101, True, source_dataloader.dataset.num_classes).to(device)
        student_model = DANN_GRL.DANN_GRL_Resnet(ResNet.resnet34, student_pretrained, source_dataloader.dataset.num_classes).to(device)

    distil = distiller.Distiller(teacher_model, student_model)
    distil = distil.to(device)
    criterion = nn.CrossEntropyLoss()

    growth_rate = torch.log(torch.FloatTensor([end_beta / init_beta])) / torch.FloatTensor([epochs])


    optimizer_da = torch.optim.SGD(distil.t_net.parameters(), init_lr_da,
                                momentum=momentum, weight_decay=weight_decay)

    optimizer_kd = torch.optim.SGD(list(distil.s_net.parameters()) + list(distil.Connectors.parameters()), init_lr_kd,
                                momentum=momentum, weight_decay=weight_decay)

    mmd_hinton_alt(init_lr_da, init_lr_kd, device, epochs, alpha, gamma, growth_rate, init_beta, source_dataloader,
               target_dataloader, target_testloader, optimizer_da, optimizer_kd, distil, criterion, is_scheduler_kd=False, is_scheduler_da=True, is_debug=False, best_teacher_acc = 0)


if __name__ == "__main__":
    main()
