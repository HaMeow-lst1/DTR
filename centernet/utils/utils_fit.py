import os

import torch
from nets.centernet_training import focal_loss, reg_l1_loss
from tqdm import tqdm

from utils.utils import get_lr
import DTR_config
'''
由于本方法只在resnet50上面实验，因此下面if else我就直接修改了
'''


def fit_one_epoch_stage1(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, backbone, save_period, save_dir, local_rank=0):
    total_r_loss    = 0
    total_c_loss    = 0
    total_loss      = 0
    val_loss        = 0
    
    dtr = DTR_config.DTR

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
        batch_thermal_images, batch_visibleBig_images, batch_visibleSmall_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        loss            = 0.0
        if not fp16:
            #thermal
            hm, wh, offset  = model_train(batch_thermal_images)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
            loss            += (c_loss + wh_loss + off_loss) * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)

            total_loss      += loss.item() * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)
            total_c_loss    += c_loss.item() * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)
            total_r_loss    += (wh_loss.item() + off_loss.item()) * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)
            
            #visibleBig
            hm, wh, offset  = model_train(batch_visibleBig_images)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
            loss            += (c_loss + wh_loss + off_loss) * dtr.lambda1_visibleBig

            total_loss      += loss.item() * dtr.lambda1_visibleBig
            total_c_loss    += c_loss.item() * dtr.lambda1_visibleBig
            total_r_loss    += (wh_loss.item() + off_loss.item()) * dtr.lambda1_visibleBig
            
            #visibleSmall
            hm, wh, offset  = model_train(batch_visibleSmall_images)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
            loss            += (c_loss + wh_loss + off_loss) * dtr.lambda1_visibleSmall

            total_loss      += loss.item() * dtr.lambda1_visibleSmall
            total_c_loss    += c_loss.item() * dtr.lambda1_visibleSmall
            total_r_loss    += (wh_loss.item() + off_loss.item()) * dtr.lambda1_visibleSmall

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #thermal
                hm, wh, offset  = model_train(batch_thermal_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    
                loss            = (c_loss + wh_loss + off_loss) * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)

                total_loss      += loss.item() * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)
                total_c_loss    += c_loss.item() * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)
                total_r_loss    += (wh_loss.item() + off_loss.item()) * (1 - dtr.lambda1_visibleBig - dtr.lambda1_visibleSmall)
                
                #visibleBig
                hm, wh, offset  = model_train(batch_visibleBig_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    
                loss            = (c_loss + wh_loss + off_loss) * dtr.lambda1_visibleBig

                total_loss      += loss.item() * dtr.lambda1_visibleBig
                total_c_loss    += c_loss.item() * dtr.lambda1_visibleBig
                total_r_loss    += (wh_loss.item() + off_loss.item()) * dtr.lambda1_visibleBig
                
                #visibleSmall
                hm, wh, offset  = model_train(batch_visibleSmall_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    
                loss            = (c_loss + wh_loss + off_loss) * dtr.lambda1_visibleSmall

                total_loss      += loss.item() * dtr.lambda1_visibleSmall
                total_c_loss    += c_loss.item() * dtr.lambda1_visibleSmall
                total_r_loss    += (wh_loss.item() + off_loss.item()) * dtr.lambda1_visibleSmall
                

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                'total_c_loss'  : total_c_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
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
            
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, _, __, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            if backbone=="resnet50":
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss            = c_loss + wh_loss + off_loss

                val_loss        += loss.item()
            else:
                outputs = model_train(batch_images)
                index   = 0
                loss    = 0
                for output in outputs:
                    hm, wh, offset  = output["hm"].sigmoid(), output["wh"], output["reg"]
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            += c_loss + wh_loss + off_loss
                    index           += 1
                val_loss            += loss.item() / index

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        
        







def fit_one_epoch_stage2(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, backbone, save_period, save_dir, local_rank=0):
    total_r_loss    = 0
    total_c_loss    = 0
    total_loss      = 0
    val_loss        = 0
    
    dtr = DTR_config.DTR

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
        batch_thermal_images, _, batch_visibleSmall_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
        batch_size = batch_thermal_images.shape[0]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        loss            = 0.0
        if not fp16:
            #thermal
            [(hm, wh, offset), feature_thermal]  = model_train(batch_thermal_images, stage = 2)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
            loss            += (c_loss + wh_loss + off_loss) * (1 - dtr.lambda2_visibleSmall)

            total_loss      += loss.item() * (1 - dtr.lambda2_visibleSmall)
            total_c_loss    += c_loss.item() * (1 - dtr.lambda2_visibleSmall)
            total_r_loss    += (wh_loss.item() + off_loss.item()) * (1 - dtr.lambda2_visibleSmall)
            
            #visibleSmall
            [(hm, wh, offset), feature_visibleSmall]  = model_train(batch_visibleSmall_images, stage = 2)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
            loss            += (c_loss + wh_loss + off_loss) * dtr.lambda2_visibleSmall

            total_loss      += loss.item() * dtr.lambda2_visibleSmall
            total_c_loss    += c_loss.item() * dtr.lambda2_visibleSmall
            total_r_loss    += (wh_loss.item() + off_loss.item()) * dtr.lambda2_visibleSmall
            
            
            loss_fd = torch.mean(torch.pow(feature_thermal - feature_visibleSmall, 2)) * batch_size
            loss += loss_fd * dtr.lambda_disentangle

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #thermal
                [(hm, wh, offset), feature_thermal]  = model_train(batch_thermal_images, stage = 2)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    
                loss            = (c_loss + wh_loss + off_loss) * (1 - dtr.lambda2_visibleSmall)

                total_loss      += loss.item() * (1 - dtr.lambda2_visibleSmall)
                total_c_loss    += c_loss.item() * (1 - dtr.lambda2_visibleSmall)
                total_r_loss    += (wh_loss.item() + off_loss.item()) * (1 - dtr.lambda2_visibleSmall)
                
                #visibleSmall
                [(hm, wh, offset), feature_visibleSmall]  = model_train(batch_visibleSmall_images, stage = 2)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    
                loss            = (c_loss + wh_loss + off_loss) * dtr.lambda2_visibleSmall

                total_loss      += loss.item() * dtr.lambda2_visibleSmall
                total_c_loss    += c_loss.item() * dtr.lambda2_visibleSmall
                total_r_loss    += (wh_loss.item() + off_loss.item()) * dtr.lambda2_visibleSmall
                
                
                
                loss_fd = torch.mean(torch.pow(feature_thermal - feature_visibleSmall, 2)) * batch_size
                loss += loss_fd * dtr.lambda_disentangle
                
                

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                'total_c_loss'  : total_c_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
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
            
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, _, __, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            if backbone=="resnet50":
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss            = c_loss + wh_loss + off_loss

                val_loss        += loss.item()
            else:
                outputs = model_train(batch_images)
                index   = 0
                loss    = 0
                for output in outputs:
                    hm, wh, offset  = output["hm"].sigmoid(), output["wh"], output["reg"]
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            += c_loss + wh_loss + off_loss
                    index           += 1
                val_loss            += loss.item() / index

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))







def fit_one_epoch_stage3(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, backbone, save_period, save_dir, local_rank=0):
    total_r_loss    = 0
    total_c_loss    = 0
    total_loss      = 0
    val_loss        = 0
    
    dtr = DTR_config.DTR

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
        batch_thermal_images, batch_visibleBig_images, _, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
        batch_size = batch_thermal_images.shape[0]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        loss            = 0.0
        if not fp16:
            #thermal
            [(hm, wh, offset), feature_thermal]  = model_train(batch_thermal_images, stage = 3)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
            loss            += (c_loss + wh_loss + off_loss) * (1 - dtr.lambda3_visibleBig)

            total_loss      += loss.item() * (1 - dtr.lambda3_visibleBig)
            total_c_loss    += c_loss.item() * (1 - dtr.lambda3_visibleBig)
            total_r_loss    += (wh_loss.item() + off_loss.item()) * (1 - dtr.lambda3_visibleBig)
            
            #visibleBig
            [(hm, wh, offset), feature_visibleBig]  = model_train(batch_visibleBig_images, stage = 3)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                
            loss            += (c_loss + wh_loss + off_loss) * dtr.lambda3_visibleBig

            total_loss      += loss.item() * dtr.lambda3_visibleBig
            total_c_loss    += c_loss.item() * dtr.lambda3_visibleBig
            total_r_loss    += (wh_loss.item() + off_loss.item()) * dtr.lambda3_visibleBig
            
            
            loss_fr = torch.mean(torch.pow(feature_thermal - feature_visibleBig, 2)) * batch_size
            loss += loss_fr * dtr.lambda_restore

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #thermal
                [(hm, wh, offset), feature_thermal]  = model_train(batch_thermal_images, stage = 3)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    
                loss            = (c_loss + wh_loss + off_loss) * (1 - dtr.lambda3_visibleBig)

                total_loss      += loss.item() * (1 - dtr.lambda3_visibleBig)
                total_c_loss    += c_loss.item() * (1 - dtr.lambda3_visibleBig)
                total_r_loss    += (wh_loss.item() + off_loss.item()) * (1 - dtr.lambda3_visibleBig)
                
                #visibleBig
                [(hm, wh, offset), feature_visibleBig]  = model_train(batch_visibleBig_images, stage = 3)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                    
                loss            = (c_loss + wh_loss + off_loss) * dtr.lambda3_visibleBig

                total_loss      += loss.item() * dtr.lambda3_visibleBig
                total_c_loss    += c_loss.item() * dtr.lambda3_visibleBig
                total_r_loss    += (wh_loss.item() + off_loss.item()) * dtr.lambda3_visibleBig
                
                
                
                loss_fr = torch.mean(torch.pow(feature_thermal - feature_visibleBig, 2)) * batch_size
                loss += loss_fr * dtr.lambda_restore
                
                

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                'total_c_loss'  : total_c_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
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
            
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, _, __, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            if backbone=="resnet50":
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss            = c_loss + wh_loss + off_loss

                val_loss        += loss.item()
            else:
                outputs = model_train(batch_images)
                index   = 0
                loss    = 0
                for output in outputs:
                    hm, wh, offset  = output["hm"].sigmoid(), output["wh"], output["reg"]
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            += c_loss + wh_loss + off_loss
                    index           += 1
                val_loss            += loss.item() / index

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))