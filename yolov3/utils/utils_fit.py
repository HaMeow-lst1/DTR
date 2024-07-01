import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

import DTR_config


def fit_one_epoch_stage1(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    
    dtr = DTR_config.DTR

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        thermal_images, visibleBig_images, visibleSmall_images, targets = batch[0], batch[1], batch[2], batch[3]
        with torch.no_grad():
            if cuda:
                thermal_images  = thermal_images.cuda(local_rank)
                visibleBig_images  = visibleBig_images.cuda(local_rank)
                visibleSmall_images  = visibleSmall_images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        batch_size = thermal_images.shape[0]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        loss_value = 0.0
        if not fp16:
            #----------------------#
            #   前向传播thermal
            #----------------------#
            outputs         = model_train(thermal_images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失thermal
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
                
            loss_value += loss_value_all * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)
            
            
            #----------------------#
            #   前向传播visibleBig
            #----------------------#
            outputs         = model_train(visibleBig_images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失visibleBig
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
                
            loss_value += loss_value_all * dtr.lambda1_visibleBig
            
            
            #----------------------#
            #   前向传播visibleSmall
            #----------------------#
            outputs         = model_train(visibleSmall_images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失visibleSmall
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
                
            loss_value += loss_value_all * dtr.lambda1_visibleSmall
            
            

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播thermal
                #----------------------#
                outputs         = model_train(thermal_images)

                loss_value_all  = 0
                #----------------------#
                #   计算损失thermal
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value += loss_value_all * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)
                
                #----------------------#
                #   前向传播visibleBig
                #----------------------#
                outputs         = model_train(visibleBig_images)

                loss_value_all  = 0
                #----------------------#
                #   计算损失visibleBig
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value += loss_value_all * dtr.lambda1_visibleBig
                
                #----------------------#
                #   前向传播visibleSmall
                #----------------------#
                outputs         = model_train(visibleSmall_images)

                loss_value_all  = 0
                #----------------------#
                #   计算损失visibleSmall
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value += loss_value_all * dtr.lambda1_visibleSmall

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[3]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
            loss_value  = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
 
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))






def fit_one_epoch_stage2(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    
    dtr = DTR_config.DTR

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        thermal_images, visibleSmall_images, targets = batch[0], batch[2], batch[3]
        with torch.no_grad():
            if cuda:
                thermal_images  = thermal_images.cuda(local_rank)
                visibleSmall_images  = visibleSmall_images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        batch_size = thermal_images.shape[0]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        loss_value = 0.0
        if not fp16:
            #----------------------#
            #   前向传播thermal
            #----------------------#
            [outputs, feature_thermal]         = model_train(thermal_images, stage = 2)

            loss_value_all  = 0
            #----------------------#
            #   计算损失thermal
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
                
            loss_value += loss_value_all * (1 - dtr.lambda2_visibleSmall)
            
            
            #----------------------#
            #   前向传播visibleSmall
            #----------------------#
            [outputs, feature_visibleSmall]         = model_train(visibleSmall_images, stage = 2)

            loss_value_all  = 0
            #----------------------#
            #   计算损失visibleSmall
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
                
            loss_value += loss_value_all * dtr.lambda2_visibleSmall
            
            loss_fd = torch.mean(torch.pow(feature_thermal - feature_visibleSmall, 2)) * batch_size
            loss_value += loss_fd * dtr.lambda_disentangle
            
            

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播thermal
                #----------------------#
                [outputs, feature_thermal]          = model_train(thermal_images, stage = 2)

                loss_value_all  = 0
                #----------------------#
                #   计算损失thermal
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value += loss_value_all * (1 - dtr.lambda2_visibleSmall)
                
                
                #----------------------#
                #   前向传播visibleSmall
                #----------------------#
                [outputs, feature_visibleSmall]         = model_train(visibleSmall_images, stage = 2)

                loss_value_all  = 0
                #----------------------#
                #   计算损失visibleSmall
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value += loss_value_all * dtr.lambda2_visibleSmall
                
                
                loss_fd = torch.mean(torch.pow(feature_thermal - feature_visibleSmall, 2)) * batch_size
                loss_value += loss_fd * dtr.lambda_disentangle

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[3]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
            loss_value  = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
 
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


   
def fit_one_epoch_stage3(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    
    dtr = DTR_config.DTR

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        thermal_images, visibleBig_images, targets = batch[0], batch[1], batch[3]
        with torch.no_grad():
            if cuda:
                thermal_images  = thermal_images.cuda(local_rank)
                visibleBig_images  = visibleBig_images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        batch_size = thermal_images.shape[0]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        loss_value = 0.0
        if not fp16:
            #----------------------#
            #   前向传播thermal
            #----------------------#
            [outputs, feature_thermal]         = model_train(thermal_images, stage = 3)

            loss_value_all  = 0
            #----------------------#
            #   计算损失thermal
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
                
            loss_value += loss_value_all * (1 - dtr.lambda3_visibleBig)
            
            
            #----------------------#
            #   前向传播visibleBig
            #----------------------#
            [outputs, feature_visibleBig]         = model_train(visibleBig_images, stage = 3)

            loss_value_all  = 0
            #----------------------#
            #   计算损失visibleBig
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
                
            loss_value += loss_value_all * dtr.lambda3_visibleBig
            
            loss_fr = torch.mean(torch.pow(feature_thermal - feature_visibleBig, 2)) * batch_size
            loss_value += loss_fr * dtr.lambda_restore
            
            

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播thermal
                #----------------------#
                [outputs, feature_thermal]          = model_train(thermal_images, stage = 3)

                loss_value_all  = 0
                #----------------------#
                #   计算损失thermal
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value += loss_value_all * (1 - dtr.lambda3_visibleBig)
                
                
                #----------------------#
                #   前向传播visibleBig
                #----------------------#
                [outputs, feature_visibleBig]         = model_train(visibleBig_images, stage = 3)

                loss_value_all  = 0
                #----------------------#
                #   计算损失visibleBig
                #----------------------#
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all  += loss_item
                loss_value += loss_value_all * dtr.lambda3_visibleBig
                
                
                loss_fr = torch.mean(torch.pow(feature_thermal - feature_visibleBig, 2)) * batch_size
                loss_value += loss_fr * dtr.lambda_restore

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[3]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
            loss_value  = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
 
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))