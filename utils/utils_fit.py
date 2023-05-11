import torch
from tqdm import tqdm

from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    loss        = 0
    val_loss    = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            high_rgbs, high_thermals, low_rgbs, low_thermals, targets  = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                if cuda:
                    high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor).cuda()
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor).cuda()
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor).cuda()
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor).cuda()
                    targets        = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor)
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor)
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor)
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs_high_rgb         = model_train(high_rgbs)
            #outputs_high_thermal     = model_train(high_thermals)
            outputs_low_thermal     = model_train(low_thermals)

            loss_value_all  = 0
            num_pos_all     = 0
            #----------------------#
            #   计算损失
            #----------------------#
            
            for l in range(len(outputs_high_rgb)):
                loss_item, num_pos = yolo_loss(l, outputs_high_rgb[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos
            '''   
            for l in range(len(outputs_high_thermal)):
                loss_item, num_pos = yolo_loss(l, outputs_high_thermal[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos
            '''
            for l in range(len(outputs_low_thermal)):
                loss_item, num_pos = yolo_loss(l, outputs_low_thermal[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos
            
            loss_value = loss_value_all / num_pos_all

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            high_rgbs, high_thermals, low_rgbs, low_thermals, targets  = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                if cuda:
                    #high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor).cuda()
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor).cuda()
                    #low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor).cuda()
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor).cuda()
                    targets        = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor)
                    high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor)
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor)
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                #outputs_high_rgb         = model_train(high_rgbs)
                #outputs_high_thermal     = model_train(high_thermals)
                outputs_low_thermal     = model_train(low_thermals)


                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
                '''
                for l in range(len(outputs_high_rgb)):
                    loss_item, num_pos = yolo_loss(l, outputs_high_rgb[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                
                for l in range(len(outputs_high_thermal)):
                    loss_item, num_pos = yolo_loss(l, outputs_high_thermal[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                '''
                for l in range(len(outputs_low_thermal)):
                    loss_item, num_pos = yolo_loss(l, outputs_low_thermal[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                                     
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))







def fit_one_epoch_stage1(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    loss        = 0
    val_loss    = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            high_rgbs, high_thermals, low_rgbs, low_thermals, targets  = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                if cuda:
                    high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor).cuda()
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor).cuda()
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor).cuda()
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor).cuda()
                    targets        = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor)
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor)
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor)
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs_high_rgb        = model_train(high_rgbs)[:3]
            #outputs_high_thermal     = model_train(high_thermals)
            outputs_low_rgb         = model_train(low_rgbs)[:3]
            outputs_low_thermal     = model_train(low_thermals)[:3]

            loss_value_all  = 0
            num_pos_all     = 0
            #----------------------#
            #   计算损失
            #----------------------#
            
            for l in range(len(outputs_high_rgb)):
                loss_item, num_pos = yolo_loss(l, outputs_high_rgb[l], targets)
                loss_value_all  += 0.5 * loss_item
                num_pos_all     += num_pos
            loss_value = loss_value_all / num_pos_all
            loss_value_all  = 0
            num_pos_all     = 0

            for l in range(len(outputs_low_rgb)):
                loss_item, num_pos = yolo_loss(l, outputs_low_rgb[l], targets)
                loss_value_all  += 0.5 * loss_item
                num_pos_all     += num_pos
            loss_value += loss_value_all / num_pos_all
            loss_value_all  = 0
            num_pos_all     = 0
                
            for l in range(len(outputs_low_thermal)):
                loss_item, num_pos = yolo_loss(l, outputs_low_thermal[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos
            
            loss_value += loss_value_all / num_pos_all

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            high_rgbs, high_thermals, low_rgbs, low_thermals, targets  = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                if cuda:
                    #high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor).cuda()
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor).cuda()
                    #low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor).cuda()
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor).cuda()
                    targets        = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    #high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor)
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor)
                    #low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor)
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs_low_thermal     = model_train(low_thermals)[:3]


                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#

                for l in range(len(outputs_low_thermal)):
                    loss_item, num_pos = yolo_loss(l, outputs_low_thermal[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                                     
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))







def fit_one_epoch_stage2(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, lambda_fd = 1.0):
    loss        = 0
    val_loss    = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            high_rgbs, high_thermals, low_rgbs, low_thermals, targets  = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                if cuda:
                    #high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor).cuda()
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor).cuda()
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor).cuda()
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor).cuda()
                    targets        = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    #high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor)
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor)
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor)
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            #outputs_high_rgb         = model_train(high_rgbs, True)
            out_low_rgb             = model_train(low_rgbs, True)
            out_low_thermal         = model_train(low_thermals, True)
            
            outputs_low_rgb         = out_low_rgb[:3]  
            outputs_low_thermal     = out_low_thermal[:3]

            loss_value_all  = 0
            num_pos_all     = 0
            #----------------------#
            #   计算损失
            #----------------------#
            
            for l in range(len(outputs_low_rgb)):
                loss_item, num_pos = yolo_loss(l, outputs_low_rgb[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos

            for l in range(len(outputs_low_thermal)):
                loss_item, num_pos = yolo_loss(l, outputs_low_thermal[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos
            
            loss_value = loss_value_all / num_pos_all
            
            loss_fd = torch.mean(torch.pow(out_low_rgb[-1] - out_low_thermal[-1], 2))
            
            #print('loss_yolo = {}'.format(loss_value))
            #print('loss_fd = {}'.format(loss_fd))
            
            loss_value += loss_fd * lambda_fd

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            high_rgbs, high_thermals, low_rgbs, low_thermals, targets  = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                if cuda:
                    #high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor).cuda()
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor).cuda()
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor).cuda()
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor).cuda()
                    targets        = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    #high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor)
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor)
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor)
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                out_low_rgb             = model_train(low_rgbs, True)
                out_low_thermal         = model_train(low_thermals, True)
            
                outputs_low_rgb         = out_low_rgb[:3]  
                outputs_low_thermal     = out_low_thermal[:3]


                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
            for l in range(len(outputs_low_rgb)):
                loss_item, num_pos = yolo_loss(l, outputs_low_rgb[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos

            for l in range(len(outputs_low_thermal)):
                loss_item, num_pos = yolo_loss(l, outputs_low_thermal[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos
                                     
            loss_value  = loss_value_all / num_pos_all
            
            loss_fd = torch.mean(torch.pow(out_low_rgb[-1] - out_low_thermal[-1], 2))
                       
            loss_value += loss_fd * lambda_fd
            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))






def fit_one_epoch_stage3(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, lambda_fd = 1.0, lambda_lst = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
    loss        = 0
    val_loss    = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            high_rgbs, high_thermals, low_rgbs, low_thermals, targets  = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                if cuda:
                    high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor).cuda()
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor).cuda()
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor).cuda()
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor).cuda()
                    targets        = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor)
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor)
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor)
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            out_high_rgb            = model_train(high_rgbs)
            out_low_rgb             = model_train(low_rgbs)
            out_low_thermal         = model_train(low_thermals)
            
            
            
            outputs_high_rgb        = out_high_rgb[:3]
            outputs_low_rgb         = out_low_rgb[:3]
            outputs_low_thermal     = out_low_thermal[:3]

            loss_value_all  = 0
            num_pos_all     = 0
            #----------------------#
            #   计算损失
            #----------------------#
            
            for l in range(len(outputs_high_rgb)):
                loss_item, num_pos = yolo_loss(l, outputs_high_rgb[l], targets)
                loss_value_all  += 0.5 * loss_item
                num_pos_all     += num_pos
            
            loss_value = loss_value_all / num_pos_all
            loss_value_all  = 0
            num_pos_all     = 0

            for l in range(len(outputs_low_rgb)):
                loss_item, num_pos = yolo_loss(l, outputs_low_rgb[l], targets)
                loss_value_all  += 0.5 * loss_item
                num_pos_all     += num_pos
            loss_value += loss_value_all / num_pos_all
            loss_value_all  = 0
            num_pos_all     = 0
                
            for l in range(len(outputs_low_thermal)):
                loss_item, num_pos = yolo_loss(l, outputs_low_thermal[l], targets)
                loss_value_all  += loss_item
                num_pos_all     += num_pos
            
            loss_value += loss_value_all / num_pos_all
            
            loss_fd = torch.mean(torch.pow(out_high_rgb[-1] - out_low_thermal[-1], 2))
            
            
            #print('loss_yolo = {}'.format(loss_value))
            #print('loss_fd = {}'.format(loss_fd))
            loss_value += loss_fd * lambda_fd
            '''
            for i in range(6):
                loss_feature = torch.mean(torch.abs(out_high_rgb[6 + i] - out_low_rgb[6 + i]))
                print('loss_frature_{} = {}'.format(i, loss_feature))
                loss_value += loss_feature * lambda_lst[i]
            '''

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            high_rgbs, high_thermals, low_rgbs, low_thermals, targets  = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                if cuda:
                    #high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor).cuda()
                    #high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor).cuda()
                    #low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor).cuda()
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor).cuda()
                    targets        = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    high_rgbs      = torch.from_numpy(high_rgbs).type(torch.FloatTensor)
                    high_thermals  = torch.from_numpy(high_thermals).type(torch.FloatTensor)
                    low_rgbs       = torch.from_numpy(low_rgbs).type(torch.FloatTensor)
                    low_thermals   = torch.from_numpy(low_thermals).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                #outputs_high_rgb         = model_train(high_rgbs)
                #outputs_high_thermal     = model_train(high_thermals)
                outputs_low_thermal     = model_train(low_thermals)[:3]


                loss_value_all  = 0
                num_pos_all     = 0
                #----------------------#
                #   计算损失
                #----------------------#
                '''
                for l in range(len(outputs_high_rgb)):
                    loss_item, num_pos = yolo_loss(l, outputs_high_rgb[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                
                for l in range(len(outputs_high_thermal)):
                    loss_item, num_pos = yolo_loss(l, outputs_high_thermal[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                '''
                for l in range(len(outputs_low_thermal)):
                    loss_item, num_pos = yolo_loss(l, outputs_low_thermal[l], targets)
                    loss_value_all  += loss_item
                    num_pos_all     += num_pos
                                     
                loss_value  = loss_value_all / num_pos_all

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))