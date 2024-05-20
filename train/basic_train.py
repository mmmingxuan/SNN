from torch.cuda import amp
import torch
from .utils import utils
import time
from .node import functional
import math
import os
import datetime
import json
import wandb


class train_1_loss():
    def __init__(self,model=None,device=None,lr_scheduler=None,optimizer=None,data_loader=None,data_loader_test=None,\
        loss_func=None,print_freq=100,args=None,scaler=None,epochs=320,train_sampler=None,train_tb_writer = None,te_tb_writer = None,max_test_acc1 = 0.,\
        test_acc5_at_max_test_acc1 = 0.,output_dir=None,start_epoch=0):
        super(train_1_loss, self).__init__()
        self.model=model
        self.optimizer=optimizer
        self.loss_func=loss_func
        self.data_loader=data_loader
        self.data_loader_test=data_loader_test
        self.print_freq=print_freq
        self.scaler=scaler
        self.epochs=epochs
        self.train_sampler=train_sampler
        self.device=device
        self.lr_scheduler = lr_scheduler
        self.train_tb_writer = train_tb_writer
        self.te_tb_writer = te_tb_writer
        self.max_test_acc1 = max_test_acc1
        self.test_acc5_at_max_test_acc1 = test_acc5_at_max_test_acc1 
        self.output_dir=output_dir
        self.model_without_ddp = model
        self.start_epoch=start_epoch
        self.cos_lr_T=self.epochs #Commonly we set cos_lr_T as the same as total epochs
        if type(self.device)==list:
            self.distributed=True
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=self.device)
            self.model_without_ddp = self.model.module
        else:
            self.distributed=False
            self.model.to(self.device)

    def loss_calculate(self,output,target):
        # return self.loss_func(output[-1].mean(dim=0), target) 
        return self.loss_func(output[-1][-1], target)

    def train_one_epoch(self,epoch,div=1):
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
        metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

        num_updates = epoch * len(self.data_loader)
        header = 'Epoch: [{}]'.format(epoch)

        from .record import sn 
        from .node.LIFnode import MultiStepLIFNode
        
        sn_list = [sn() for _ in range(len(MultiStepLIFNode.get_all_neurons()))]
        
        for image, target in metric_logger.log_every(self.data_loader, self.print_freq, header):
            start_time = time.time()
            image, target = image.to(self.device), target.to(self.device)
            # with torch.autograd.detect_anomaly():
            if self.scaler is not None:
                with amp.autocast():
                    output = self.model(image, target, epoch)
                    loss = torch.tensor([0.], device=output[0].device)
                    loss += self.loss_calculate(output, target)
            else:
                output = self.model(image, target, epoch)
                loss = torch.tensor([0.], device=output[0].device)
                loss += self.loss_calculate(output, target)

            self.optimizer.zero_grad()
            loss=loss/div
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                loss.backward()
                self.optimizer.step()

            from .node.LIFnode import MultiStepLIFNode
            for index, Node_instance in enumerate(MultiStepLIFNode.get_all_neurons()):
                sn_list[index].grad_1.append(Node_instance.grads_1[0])
                sn_list[index].grad_before_2.append(Node_instance.grads_before[0])        
                sn_list[index].grad_before_3.append(Node_instance.grads_before[1])
                sn_list[index].grad_before_4.append(Node_instance.grads_before[2])  
                sn_list[index].grad_after_2.append(Node_instance.grads_after[0])  
                sn_list[index].grad_after_3.append(Node_instance.grads_after[1])  
                sn_list[index].grad_after_4.append(Node_instance.grads_after[2])  
                     
            functional.reset_net(self.model)
            output = output[-1].mean(dim=0)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            loss_s = loss.item()
            if math.isnan(loss_s):
                raise ValueError('loss is Nan')
            acc1_s = acc1.item()
            acc5_s = acc5.item()
            num_updates += 1

            metric_logger.update(loss=loss_s, lr=round(float(self.optimizer.param_groups[0]["lr"]),6))

            metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
            metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
            metric_logger.meters['img/s'].update(round(float(batch_size / (time.time() - start_time)),5))
            # self.lr_scheduler.step_update(num_updates=num_updates)

        sn_log = {}
        
        import pandas as pd
        import os
        
        filename = "record_grad.csv"
        data = {}
        for index, Node_instance in enumerate(sn_list):
            Node_instance.avg()
            data[f"sn{index+1}_grad_1"] = Node_instance.avg_grad_1
            data[f"sn{index+1}_grad_before_2"] = Node_instance.avg_grad_before_2
            data[f"sn{index+1}_grad_before_3"] = Node_instance.avg_grad_before_3
            data[f"sn{index+1}_grad_before_4"] = Node_instance.avg_grad_before_4
            data[f"sn{index+1}_grad_after_2"] = Node_instance.avg_grad_after_2
            data[f"sn{index+1}_grad_after_3"] = Node_instance.avg_grad_after_3
            data[f"sn{index+1}_grad_after_4"] = Node_instance.avg_grad_after_4

        df = pd.DataFrame([data])  # 将data转换为包含一行的DataFrame
        
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False, mode='w', header=True)
        else:
            df.to_csv(filename, index=False, mode='a', header=False)

            
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, sn_log

    def evaluate_one_epoch(self):
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header='Test:'
        with torch.no_grad():
            for image, target in metric_logger.log_every(self.data_loader_test, self.print_freq, header):
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(image)
                loss=torch.tensor([0.], device=output[0].device)
                output = output[-1].mean(dim=0)
                loss += self.loss_func(output, target)
                functional.reset_net(self.model)

                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
        return loss, acc1, acc5

    def resume(self,checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        self.model_without_ddp.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.max_test_acc1 = checkpoint['max_test_acc1']
        self.test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']
        
    def save_checkpoint(self,epoch):
        checkpoint = {
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc1': self.max_test_acc1,
            'test_acc5_at_max_test_acc1': self.test_acc5_at_max_test_acc1,
        }

        utils.save_on_master(
            checkpoint,
            os.path.join(self.output_dir, 'checkpoint_latest.pth'))
        self.save_flag = False

        if epoch % 64 == 0 or epoch == self.epochs - 1:
            self.save_flag = True

        elif self.cos_lr_T == 0:
            for item in self.lr_step_size:
                if (epoch + 2) % item == 0:
                    self.save_flag = True
                    break

        if self.save_flag:
            utils.save_on_master(
                checkpoint,
                os.path.join(self.output_dir, f'checkpoint_{epoch}.pth'))

        if self.save_max:
            utils.save_on_master(
                checkpoint,
                os.path.join(self.output_dir, 'checkpoint_max_test_acc1.pth'))

    def train(self):
        start_time = time.time()
        max_test_acc = 0
        print(self.output_dir)
        for epoch in range(self.start_epoch, self.epochs):
            self.save_max = False
            # pdb.set_trace()
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            train_loss, train_acc1, train_acc5 ,sn_log = self.train_one_epoch(epoch)
            if utils.is_main_process():
                self.train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                self.train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                self.train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
            self.lr_scheduler.step()
            
            test_loss, test_acc1, test_acc5 = self.evaluate_one_epoch()
            if test_acc1 > max_test_acc: max_test_acc = test_acc1

            wandb_log = {
                "epoch":epoch,
                "train_acc1": train_acc1,
                "train_loss": train_loss,
                "test_acc1":  test_acc1,
                "test_loss": test_loss,
                "max_test_acc1": max_test_acc
            }
            # wandb_log.update(sn_log)
            wandb.log(wandb_log)
            
            
            # if self.te_tb_writer is not None:
            #     if utils.is_main_process():
            #         self.te_tb_writer.add_scalar('test_loss', test_loss, epoch)
            #         self.te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
            #         self.te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


            if self.max_test_acc1 < test_acc1:
                self.max_test_acc1 = test_acc1
                self.test_acc5_at_max_test_acc1 = test_acc5
                self.save_max = True

            if self.output_dir:
                self.save_checkpoint(epoch)
                
            # print(args)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))

            print('Training time {}'.format(total_time_str), 'max_test_acc1', self.max_test_acc1,
                'test_acc5_at_max_test_acc1', self.test_acc5_at_max_test_acc1)
        

    def train_one_epoch_save_grad(self,epoch):
        self.model.train()
        gradient_sums = {}
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
        metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

        header = 'Epoch: [{}]'.format(epoch)
        utils.mkdir(self.output_dir+"/grad")
        for image, target in metric_logger.log_every(self.data_loader, self.print_freq, header):
            start_time = time.time()
            image, target = image.to(self.device), target.to(self.device)
            # with torch.autograd.detect_anomaly():
            if self.scaler is not None:
                with amp.autocast():
                    output = self.model(image)
                    loss = torch.tensor([0.], device=output[0].device)
                    loss += self.loss_calculate(output, target)
            else:
                output = self.model(image)
                loss = torch.tensor([0.], device=output[0].device)
                loss += self.loss_calculate(output, target)


            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                loss.backward()
                self.optimizer.step()

            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         if name not in gradient_sums:
            #             gradient_sums[name] = 0.0
            #         gradient_sums[name] += param.grad.detach().cpu()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # 使用.clone()创建梯度的副本，并使用.detach()确保这些梯度不会在计算图中被跟踪
                    grad_copy = param.grad.detach().clone().cpu()
                    if name not in gradient_sums:
                        gradient_sums[name] = grad_copy
                    else:
                        gradient_sums[name] += grad_copy


            functional.reset_net(self.model)
            output = output[-1].mean(dim=0)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            loss_s = loss.item()
            if math.isnan(loss_s):
                raise ValueError('loss is Nan')
            acc1_s = acc1.item()
            acc5_s = acc5.item()

            metric_logger.update(loss=loss_s, lr=round(float(self.optimizer.param_groups[0]["lr"]),5))

            metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
            metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
            metric_logger.meters['img/s'].update(round(float(batch_size / (time.time() - start_time)),5))
        save_path = os.path.join(self.output_dir+"/grad", f"epoch_{epoch}_gradient_sums.pt")
        torch.save(gradient_sums, save_path)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
        
    def train_save_grad(self):
        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            self.save_max = False
            # pdb.set_trace()
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            train_loss, train_acc1, train_acc5 = self.train_one_epoch_save_grad(epoch)
            if utils.is_main_process():
                self.train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                self.train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                self.train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
            self.lr_scheduler.step()

            test_loss, test_acc1, test_acc5 = self.evaluate_one_epoch()
            if self.te_tb_writer is not None:
                if utils.is_main_process():
                    self.te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                    self.te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                    self.te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


            if self.max_test_acc1 < test_acc1:
                self.max_test_acc1 = test_acc1
                self.test_acc5_at_max_test_acc1 = test_acc5
                self.save_max = True

            if self.output_dir:
                self.save_checkpoint(epoch)
                
            # print(args)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(self.output_dir)

            print('Training time {}'.format(total_time_str), 'max_test_acc1', self.max_test_acc1,
                'test_acc5_at_max_test_acc1', self.test_acc5_at_max_test_acc1)
   
    def train_save_diffsum(self,div=1):
        start_time = time.time()
        self.diffsum_train={}
        self.diffsum_train_tensor={}
        self.diffsum_test={}
        self.diffsum_test_tensor={}
        os.mkdir(self.output_dir+f"/norm_diff")
        for epoch in range(self.start_epoch, self.epochs):
            self.temp={}
            self.temp_tensor={}
            self.save_max = False
            # pdb.set_trace()
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            train_loss, train_acc1, train_acc5 = self.train_one_epoch(epoch,div=div)
            self.print_temp_norm_list(self.model,reload=True)
            self.diffsum_train[epoch]=self.temp
            self.diffsum_train_tensor[epoch]=self.temp_tensor
            self.temp={}
            self.temp_tensor={}
            self.save_norm_diff_train(epoch)
            if utils.is_main_process():
                self.train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                self.train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                self.train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
            self.lr_scheduler.step(epoch+1)

            test_loss, test_acc1, test_acc5 = self.evaluate_one_epoch()
            self.print_temp_norm_list(self.model,reload=True,Train=False)
            self.diffsum_test[epoch]=self.temp
            self.diffsum_test_tensor[epoch]=self.temp_tensor
            self.temp={}
            self.temp_tensor={}
            wandb.log({"epoch":epoch, "train_acc1": train_acc1, "train_loss": train_loss, "test_acc1":  test_acc1, "test_loss": test_loss})
            if self.te_tb_writer is not None:
                if utils.is_main_process():
                    self.te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                    self.te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                    self.te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


            if self.max_test_acc1 < test_acc1:
                self.max_test_acc1 = test_acc1
                self.test_acc5_at_max_test_acc1 = test_acc5
                self.save_max = True

            if self.output_dir:
                self.save_checkpoint(epoch)
                
            # print(args)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(self.output_dir)

            print('Training time {}'.format(total_time_str), 'max_test_acc1', self.max_test_acc1,
                'test_acc5_at_max_test_acc1', self.test_acc5_at_max_test_acc1)
            self.save_norm_diff_test(epoch)

    def save_norm_diff_train(self,epoch):
        with open(self.output_dir+f"/norm_diff/norm_diff_{epoch}_train.json","w") as f:
            json.dump(self.diffsum_train,f)
        if os.path.exists(self.output_dir+f"/norm_diff/norm_diff_{epoch-1}_train.json"):
            os.remove(self.output_dir+f"/norm_diff/norm_diff_{epoch-1}_train.json")
            
        torch.save(self.diffsum_train_tensor,self.output_dir+f"/norm_diff/norm_diff_{epoch}_train.pt")
        if os.path.exists(self.output_dir+f"/norm_diff/norm_diff_{epoch-1}_train.pt"):
            os.remove(self.output_dir+f"/norm_diff/norm_diff_{epoch-1}_train.pt")

    def save_norm_diff_test(self,epoch):
        with open(self.output_dir+f"/norm_diff/norm_diff_{epoch}_test.json","w") as f:
            json.dump(self.diffsum_test,f)
        if os.path.exists(self.output_dir+f"/norm_diff/norm_diff_{epoch-1}_test.json"):
            os.remove(self.output_dir+f"/norm_diff/norm_diff_{epoch-1}_test.json")

        torch.save(self.diffsum_train_tensor,self.output_dir+f"/norm_diff/norm_diff_{epoch}_test.pt")
        if os.path.exists(self.output_dir+f"/norm_diff/norm_diff_{epoch-1}_test.pt"):
            os.remove(self.output_dir+f"/norm_diff/norm_diff_{epoch-1}_test.pt")


    def print_temp_norm_list(self,module, parent_name='',reload=False,Train=True):
        """
        递归遍历模型的所有子模块，并打印含有temp_norm_list属性的模块的名称和值。
        """
        for name, child in module.named_children():
            # 构建当前层的完整路径名称
            if parent_name:
                current_name = f"{parent_name}.{name}"
            else:
                current_name = name

            # 如果当前子模块有temp_norm_list属性，打印它
            if hasattr(child, 'temp_norm_list'):
                # print(f"Path: {current_name}, Value: {getattr(child, 'temp_norm_list')}")
                if current_name not in self.temp:
                    self.temp[current_name]={}
                self.temp[current_name]['temp_norm_list']=getattr(child, 'temp_norm_list')
                if reload:
                    child.temp_norm_list=[]

            if hasattr(child, 'grad_l_1_list'):
                # print(f"Path: {current_name}, Value: {getattr(child, 'grad_l_1_list')}")
                if current_name not in self.temp:
                    self.temp[current_name]={}
                if Train:
                    self.temp[current_name]['grad_l_1_list']=[[float(l.grad) for l in lists]for lists in getattr(child, 'grad_l_1_list')]
                if reload:
                    child.grad_l_1_list=[]

            # if hasattr(child, 'grad_l_list'):
            #     # print(f"Path: {current_name}, Value: {getattr(child, 'grad_l_list')}")
            #     if current_name not in self.temp:
            #         self.temp[current_name]={}
            #     if Train:
            #         self.temp[current_name]['grad_l_list']=getattr(child, 'grad_l_list')
            #     if reload:
            #         child.grad_l_list=[]

            if hasattr(child, 'grad_l_list'):
                # print(f"Path: {current_name}, Value: {getattr(child, 'grad_l_list')}")
                if current_name not in self.temp:
                    self.temp[current_name]={}
                if Train:
                    self.temp[current_name]['grad_l_list']=[[float(l.grad) for l in lists]for lists in getattr(child, 'grad_l_list')]
                if reload:
                    child.grad_l_list=[]

            if hasattr(child, 'delta_v'):
                # print(f"Path: {current_name}, Value: {getattr(child, 'grad_l_list')}")
                if current_name not in self.temp_tensor:
                    self.temp_tensor[current_name]={}
                if Train:
                    self.temp_tensor[current_name]['delta_v']=getattr(child, 'delta_v')
                if reload:
                    child.delta_v=[]

            # 递归检查子模块
            self.print_temp_norm_list(child, current_name,reload,Train)

class train_4_loss(train_1_loss):
    def loss_calculate(self,output,target):
        loss=0
        for i in range(len(output)): 
            loss=loss+self.loss_func(output[i].mean(dim=0), target) 
        return loss

class train_4_TET_loss(train_1_loss):
    def loss_calculate(self,output,target):
        loss=0
        for i in range(len(output[-1])): 
            loss=loss+self.loss_func(output[-1][i], target) 
        return loss
    
class train_4_ver_TET_loss(train_1_loss):
    def loss_calculate(self,output,target):
        loss=0
        for i in range(len(output)): 
            loss=loss+self.loss_func(output[i][-1], target) 
        return loss

class train_16_loss(train_1_loss):
    def loss_calculate(self,output,target):
        loss=0
        for i in range(len(output)): 
            for j in range(len(output[0])):
                loss=loss+self.loss_func(output[i][j], target) 
        return loss

class train_16_loss_high_zhanbi_1(train_1_loss):
    #output.size()=[4,4,128,100],target.size()=[128]
    def loss_calculate(self,output,target):
        loss=0
        confidences = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        weight = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp_output = output[i][j].detach()
                probabilities = torch.nn.functional.softmax(temp_output, dim=1)
                confidence = probabilities[torch.arange(probabilities.size(0)), target]
                confidences[i][j] = confidence

        for i in range(len(output)):
            stage_sum = sum(confidences[i]) 
            for j in range(len(output[i])):
                weight[i][j] = confidences[i][j] / stage_sum

        for i in range(len(output)):
            for j in range(len(output[i])):
                avg_weight = torch.mean(weight[i][j])
                loss += self.loss_func(output[i][j], target) * avg_weight

        return loss

class train_16_loss_high_zhanbi_2(train_1_loss):
    #output.size()=[4,4,128,100],target.size()=[128]
    def loss_calculate(self,output,target):
        loss=0
        confidences = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        weight = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp_output = output[i][j].detach()
                probabilities = torch.nn.functional.softmax(temp_output, dim=1)
                confidence = probabilities[torch.arange(probabilities.size(0)), target]
                confidences[i][j] = confidence.mean()

        for i in range(len(output)):
            stage_sum = sum(confidences[i]) 
            for j in range(len(output[i])):
                weight[i][j] = confidences[i][j] / stage_sum

        for i in range(len(output)):
            for j in range(len(output[i])):
                loss += self.loss_func(output[i][j], target) * weight[i][j] 

        return loss


class train_16_loss_high_softmax_3(train_1_loss):
    #output.size()=[4,4,128,100],target.size()=[128]
    def loss_calculate(self,output,target):
        loss=0
        confidences = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        weight = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp_output = output[i][j].detach()
                probabilities = torch.nn.functional.softmax(temp_output, dim=1)
                confidence = probabilities[torch.arange(probabilities.size(0)), target]
                confidences[i][j] = confidence

        for i in range(len(output)):
            stage_sum = sum(confidences[i]) 
            for j in range(len(output[i])):
                #改写为softmax
                weight[i][j] = confidences[i][j] / stage_sum

        for i in range(len(output)):
            for j in range(len(output[i])):
                avg_weight = torch.mean(weight[i][j])
                loss += self.loss_func(output[i][j], target) * avg_weight
        return loss


class train_16_loss_high_zhanbi_4(train_1_loss):
    #output.size()=[4,4,128,100],target.size()=[128]
    def loss_calculate(self,output,target):
        loss=0
        confidences = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        weight = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp_output = output[i][j].detach()
                probabilities = torch.nn.functional.softmax(temp_output, dim=1)
                confidence = probabilities[torch.arange(probabilities.size(0)), target]
                confidences[i][j] = confidence

        for i in range(len(output)):
            stage_sum = sum(confidences[i]) 
            for j in range(len(output[i])):
                weight[i][j] = confidences[i][j] / stage_sum

        for i in range(len(output)):
            for j in range(len(output[i])):
                avg_weight = torch.mean(weight[i][j])
                loss += self.loss_func(output[i][j], target) * avg_weight * 4

        return loss


class train_16_loss_low_zhanbi_5(train_1_loss):
    #output.size()=[4,4,128,100],target.size()=[128]
    def loss_calculate(self,output,target):
        loss=0
        confidences = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        weight = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp_output = output[i][j].detach()
                probabilities = torch.nn.functional.softmax(temp_output, dim=1)
                confidence = probabilities[torch.arange(probabilities.size(0)), target]
                confidences[i][j] = confidence

        for i in range(len(output)):
            stage_sum = sum(confidences[i]) 
            for j in range(len(output[i])):
                weight[i][j] = confidences[i][j] / stage_sum

        for i in range(len(output)):
            for j in range(len(output[i])):
                avg_weight = torch.mean(weight[i][j])
                loss += self.loss_func(output[i][j], target) * (1.0/avg_weight)

        return loss

class train_16_loss_low_zhanbi_8(train_1_loss):
    #output.size()=[4,4,128,100],target.size()=[128]
    def loss_calculate(self,output,target):
        loss=0
        confidences = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        weight = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp_output = output[i][j].detach()
                probabilities = torch.nn.functional.softmax(temp_output, dim=1)
                confidence = probabilities[torch.arange(probabilities.size(0)), target]
                confidences[i][j] = confidence

        for i in range(len(output)):
            stage_sum = sum(confidences[i]) 
            for j in range(len(output[i])):
                weight[i][j] = confidences[i][j] / stage_sum

        for i in range(len(output)):
            for j in range(len(output[i])):
                avg_weight = torch.mean(weight[i][j])
                loss += self.loss_func(output[i][j], target) * (1.0/(avg_weight * 4))

        return loss

class train_16_loss_low_zhanbi_9(train_1_loss):
    #output.size()=[4,4,128,100],target.size()=[128]
    def loss_calculate(self,output,target):
        loss=0
        confidences = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        weight = [[None for _ in range(len(output[0]))] for _ in range(len(output))]
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp_output = output[i][j].detach()
                probabilities = torch.nn.functional.softmax(temp_output, dim=1)
                confidence = probabilities[torch.arange(probabilities.size(0)), target]
                confidences[i][j] = confidence

        for i in range(len(output)):
            stage_sum = sum(confidences[i]) 
            for j in range(len(output[i])):
                weight[i][j] = confidences[i][j] / stage_sum

        for i in range(len(output)):
            for j in range(len(output[i])):
                avg_weight = torch.mean(weight[i][j])
                loss += self.loss_func(output[i][j], target) * (1.0-avg_weight)

        return loss

train_dict={
    "1":train_1_loss,
    "4":train_4_loss,
    "TET":train_4_TET_loss,
    "rTET":train_4_ver_TET_loss,
    "16":train_16_loss,
    "16-1":train_16_loss_high_zhanbi_1,
    "16-2":train_16_loss_high_zhanbi_2,
    "16-3":train_16_loss_high_softmax_3,
    "16-4":train_16_loss_high_zhanbi_4,
    "16-5":train_16_loss_low_zhanbi_5,
    "16-8":train_16_loss_low_zhanbi_8,
    "16-9":train_16_loss_low_zhanbi_9,
}

def init_train(train):
    return train_dict[train]