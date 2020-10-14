import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import eval, adjust_learning_rate
#from DAN_Model import DANNet_ResNet
import distiller
import DA.DA_datasets as DA_datasets
import cmodels.ResNet as ResNet
#import cmodels.DAN_model as DAN_model
import cmodels.DANN_GRL as DANN_GRL
#import cmodels.alexnet as alexnet
from utils import eval, adjust_learning_rate, get_sub_dataset_name
import os

def od_mmd_one_epoch(current_ep, epochs, distils, source_dataloader, targets_dataloader, optimizer_das, optimizer_kds, criterion, device, alpha, beta, gamma, batch_norm, is_cst):
	if batch_norm:
		for distil in distils:
			distil.train()
			distil.s_net.train()
			distil.t_net.train()

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
		data_source, label_source = data_source.to(device), label_source.to(device)

		for ix, it in enumerate(iter_targets):

			try:
				data_target, _ = it.next()
			except StopIteration:
				it = iter(targets_dataloader[ix])
				data_target, _ = it.next()

			if data_target.shape[0] != data_source.shape[0]:
				data_target = data_target[: data_source.shape[0]]
			data_target = data_target.to(device)

			#Teacher domain adaption

			optimizer_das[ix].zero_grad()
			p = float(i + (current_ep -1) * len(source_dataloader)) / epochs / len(source_dataloader)
			delta = 2. / (1. + np.exp(-10 * p)) - 1
			teacher_label_source_pred, teacher_source_loss_adv = distils[ix].t_net(data_source, delta)
			teacher_source_loss_cls = F.cross_entropy(F.log_softmax(teacher_label_source_pred, dim=1), label_source)

			_, teacher_target_loss_adv = distils[ix].t_net(data_target, delta, source=False)
			teacher_loss_adv = teacher_source_loss_adv + teacher_target_loss_adv

			teacher_da_grl_loss = (1 - beta) * (teacher_source_loss_cls + gamma * teacher_loss_adv)
			teacher_da_temp_losses[ix] += teacher_da_grl_loss.mean().item()

			teacher_da_grl_loss.mean().backward()
			optimizer_das[ix].step()
			optimizer_das[ix].zero_grad()

			#Knowledge Distillation
		
			optimizer_kds[ix].zero_grad()
			_, distil_loss_source = distils[ix](data_source)
			if torch.cuda.device_count() > 1:
				source_outputs_distillar = distils[ix].s_net.module.nforward(data_source)
			else:
				source_outputs_distillar = distils[ix].s_net.nforward(data_source)

			source_outputs_distillar = F.log_softmax(source_outputs_distillar, dim=1)

			_, distil_loss_target = distils[ix](data_target)
			_, student_target_loss_adv = distils[ix].s_net(data_target, delta, source=False)

			class_loss = criterion(source_outputs_distillar, label_source)
			kd_loss = beta * (class_loss + distil_loss_target.sum()/data_source.shape[0]/10000 + distil_loss_source.sum() / data_target.shape[0] / 10000 + alpha * student_target_loss_adv)
			kd_temp_losses[ix] += kd_loss.mean().item()

			total_losses[ix] += teacher_da_grl_loss.mean().item() + kd_loss.mean().item()

			kd_loss.mean().backward()
			optimizer_kds[ix].step()
			optimizer_kds[ix].zero_grad()
		
	return total_losses / len(source_dataloader), total_losses / len(source_dataloader), teacher_da_temp_losses / len(source_dataloader)


def od_mmd_train(init_lr_da, init_lr_kd, epochs, growth_rate, alpha, gamma, init_beta, distils, source_dataloader, targets_dataloader, targets_testloader,
	optimizer_das, optimizer_kds, criterion, device, batch_norm, is_scheduler_da=True, is_scheduler_kd=True, scheduler_da=None, scheduler_kd=None, is_cst=True):
	
	total_loss_arr = []
	teacher_da_temp_loss_arr = []
	kd_temp_loss_arr = []
	teacher_target_acc_arr = []
	student_target_acc_arr = []

	best_student_acc = 0.
	best_teacher_acc = 0.
	epochs += 1
	for epoch in range(1, epochs):
		beta = init_beta * torch.exp(growth_rate * (epoch - 1))
		beta = beta.to(device)

		if (is_scheduler_da):
			new_lr_da = init_lr_da / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
			for optimizer_da in optimizer_das:
				adjust_learning_rate(optimizer_da, new_lr_da)

		if (is_scheduler_kd):
			new_lr_kd = init_lr_kd / np.power((1 + 10 * (epoch - 1) / epochs), 0.75) # 10*
			for optimizer_kd in optimizer_kds:
				adjust_learning_rate(optimizer_kd, new_lr_kd)

		total_loss_1, total_loss_2, teacher_da_temp_loss_1 = od_mmd_one_epoch(epoch, epochs, distils, source_dataloader,
											targets_dataloader, optimizer_das, optimizer_kds,
											criterion, device,alpha, beta, gamma, batch_norm, is_cst)

		students_targets_acc = np.zeros(len(distils))

		for i, d in enumerate(targets_testloader):
			students_targets_acc[i] = eval(distils[i].s_net, device, d, False)

		total_target_acc = students_targets_acc.mean()
		print(f'epoch : {epoch}\tacc : {total_target_acc}')

		if (total_target_acc > best_student_acc):
			best_student_acc = total_target_acc

			torch.save({'student_model': distils[0].s_net.state_dict(), 'acc': best_student_acc, 'epoch': epoch},
		               "./student_model.pth")

		if scheduler_da is not None:
			scheduler_da.step()

		if scheduler_kd is not None:
			scheduler_kd.step()

		if(epoch == 150 and epoch == 250):
			for optimizer_kd in optimizer_kds:
				for param_group in optimizer_kd.param_groups:
					param_group['lr'] = param_group['lr'] * .1
	return best_student_acc


def main():
	batch_size = 32
	test_batch_size = 16

	a = os.path.expanduser('../../datasets/Art')
	c = os.path.expanduser('../../datasets/Clipart')
	p = os.path.expanduser('../../datasets/Product')
	r = os.path.expanduser('../../datasets/RealWorld')
	dataset_name = 'Office31'
	source_dataset_path = a
	target_dataset_path_1 = c
	target_dataset_path_2 = p
	target_dataset_path_3 = r
	
	batch_norm = True
	epochs = 400
	init_lr_da = 0.0001
	init_lr_kd = 0.001
	momentum = 0.9
	weight_decay = 5e-4
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	alpha = 0.5
	gamma = 0.5
	init_beta = 0.1
	end_beta = 0.8
	is_scheduler_da = True
	is_scheduler_kd = False

	source_dataloader, targets_dataloader, targets_testloader = DA_datasets.get_source_m_target_loader(dataset_name,
													source_dataset_path,
													[target_dataset_path_1, target_dataset_path_2,target_dataset_path_3],
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

	distil_1 = distiller.Distiller(teacher_model_1, student_model)
	distil_1 = distil_1.to(device)
	distil_2 = distiller.Distiller(teacher_model_2, student_model)
	distil_2 = distil_2.to(device)
	distil_3 = distiller.Distiller(teacher_model_3, student_model)
	distil_3 = distil_3.to(device)

	distils = [distil_1, distil_2, distil_3]

	criterion = nn.CrossEntropyLoss()
	growth_rate = torch.zeros(1)
	if init_beta != 0.0:
		growth_rate = torch.log(torch.FloatTensor([end_beta / init_beta])) / torch.FloatTensor([epochs])

	optimizer_da_1 = torch.optim.SGD(list(distil_1.s_net.parameters()) + list(distil_1.t_net.parameters()), init_lr_da,
					momentum=momentum, weight_decay=weight_decay)

	optimizer_da_2 = torch.optim.SGD(list(distil_2.s_net.parameters()) + list(distil_2.t_net.parameters()), init_lr_da,
					momentum=momentum, weight_decay=weight_decay)

	optimizer_da_3 = torch.optim.SGD(list(distil_3.s_net.parameters()) + list(distil_3.t_net.parameters()), init_lr_da,
					momentum=momentum, weight_decay=weight_decay)

	optimizer_kd_1 = torch.optim.SGD(list(distil_1.s_net.parameters()) + list(distil_1.Connectors.parameters()), init_lr_kd,
					momentum=momentum, weight_decay=weight_decay)

	optimizer_kd_2 = torch.optim.SGD(list(distil_2.s_net.parameters()) + list(distil_2.Connectors.parameters()), init_lr_kd,
					momentum=momentum, weight_decay=weight_decay)

	optimizer_kd_3 = torch.optim.SGD(list(distil_3.s_net.parameters()) + list(distil_3.Connectors.parameters()), init_lr_kd,
					momentum=momentum, weight_decay=weight_decay)
	
	optimizer_kds = [optimizer_kd_1, optimizer_kd_2, optimizer_kd_3]
	optimizer_das = [optimizer_da_1, optimizer_da_2, optimizer_da_3]


	od_mmd_train(init_lr_da, init_lr_kd, epochs, growth_rate, alpha, gamma, init_beta, distils, source_dataloader, targets_dataloader, targets_testloader,
		optimizer_das, optimizer_kds, criterion, device, batch_norm, is_scheduler_da = is_scheduler_da, is_scheduler_kd = is_scheduler_kd, scheduler_da=None, scheduler_kd=None)

if __name__ == "__main__":
	main()

