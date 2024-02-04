from loguru import logger
import time
import copy
import datetime
from common.func import *


def exp_cross_session(subject, losser, save_path, model_name,
                      frist_epoch, eary_stop_epoch, second_epoch,
                      kfolds, batch_size, device,
                      x_train, y_train, x_test, y_test,
                      nChan=22, nTime=1000, nClass=4):
    logger.info('\nTraining on subject ' + str(subject) + '\n')
    avg_eval_acc = 0
    accuracy_val_list1 = []
    accuracy_val_list2 = []
    kappa_val_list1 = []
    kappa_val_list2 = []
    y_train = y_train.reshape(-1, 1)
    for kfold, (train_dataset, valid_dataset, split_train_index, split_validation_index) in enumerate(
            cross_validate(x_train, y_train, kfolds)):

        info = 'Subject_{}_fold_{}:'.format(subject, kfold)
        logger.info(info + '\n')

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化
        model = getModel(model_name=model_name, device=device, nChan=nChan, nTime=nTime, nClass=nClass)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3)

        if model_name == "Conformer":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # scheduler = None # torch.optim.lr_scheduler.StepLR(optimizer,gamma=0.99,step_size=10)
        # scheduler_adap = None

        ### First step
        best_loss_kfold = np.inf
        best_loss_kfold_acc = 0
        best_acc_kfold = 0
        best_acc_kfold_loss = np.inf
        mini_loss = None
        remaining_epoch = eary_stop_epoch
        for iter in range(frist_epoch):
            loss_train = 0
            accuracy_train = 0
            model.train()
            for inputs, target, index in train_dataloader:
                target = target[:, 0]
                a = target.size()[0]
                b = target.size()[0]
                radio = 1.0 * b / a
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()  # 清空梯度
                output = model(inputs)
                loss = losser(output, target)
                accuracy_train += torch.sum(torch.argmax(output, dim=1) == target, dtype=torch.float32) / (
                        radio * len(train_dataset))
                loss_train += loss.detach().item() / len(train_dataloader)
                loss.backward()  # 反向传播和计算梯度
                optimizer.step()  # 更新参数
            loss_val, accuracy_val, confusion_val, kappa_val = validate_model(model, valid_dataset, device, losser,
                                                                              n_calsses=nClass)
            remaining_epoch = remaining_epoch - 1

            if remaining_epoch <= 0:
                avg_eval_acc += best_acc_kfold
                break
            if mini_loss is None or loss_train < mini_loss:
                mini_loss = loss_train

            if loss_val < best_loss_kfold:
                if accuracy_val >= best_acc_kfold:
                    best_model = copy.deepcopy(model.state_dict())
                    optimizer_state = copy.deepcopy(optimizer.state_dict())
                    best_acc_kfold = accuracy_val
                    best_acc_kfold_loss = loss_val
                remaining_epoch = eary_stop_epoch
                best_loss_kfold = loss_val
                best_loss_kfold_acc = accuracy_val

            if accuracy_val > best_acc_kfold:
                best_model = copy.deepcopy(model.state_dict())
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_acc_kfold = accuracy_val
                best_acc_kfold_loss = loss_val
                remaining_epoch = eary_stop_epoch

            info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = info + '\tKfold:{0:1}\tEpoch:{1:3}\tTra_Loss:{2:.3}\tTr_acc:{3:.3}\tVa_Loss:{4:.3}\tVa_acc:{5:.3}\tMinVloss:{6:.3}\tToacc:{7:.3}\tMaxVacc:{8:.3}\tToloss:{9:.3}\tramainingEpoch:{10:3}' \
                .format(kfold + 1, iter, loss_train, accuracy_train, loss_val, accuracy_val, best_loss_kfold,
                        best_loss_kfold_acc, best_acc_kfold, best_acc_kfold_loss, remaining_epoch)
            logger.info('\n' + info + '\n')

        info = f'Early stopping at Epoch {iter},and retrain the Net using both the training data and validation data.'
        print(info)
        logger.info(info + '\n')

        ### Second step
        model.load_state_dict(best_model)
        optimizer.load_state_dict(optimizer_state)

        for iter in range(second_epoch):
            model.train()
            x_train_all, y_train_all = torch.FloatTensor(x_train), torch.LongTensor(y_train)
            train_index = torch.LongTensor([i for i, _ in enumerate(y_train_all)])
            train_dataset_all = TensorDataset(x_train_all, y_train_all, train_index)
            train_dataloader_all = DataLoader(train_dataset_all, batch_size=batch_size, shuffle=True)

            for inputs, target, index in train_dataloader_all:
                inputs = inputs.to(device)
                target = target[:, 0]
                target = target.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss = losser(output, target)
                loss.backward()
                optimizer.step()
            loss_val, accuracy_val, confusion_val, kappa_val = validate_model(model, valid_dataset, device, losser,
                                                                              n_calsses=nClass)

            info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            info = info + '\tKfold:{0:1}\tEpoch:{1:3}\tVa_Loss:{2:.3}\tVa_acc:{3:.3}' \
                .format(kfold + 1, iter, loss_val, accuracy_val)
            logger.info('\n' + info + '\n')
            if loss_val < mini_loss:
                break

        # save model.
        file_name1 = '{}_sub{}_fold{}_acc{:.4}_best_model_step1.pth'.format(model_name, subject, kfold, best_acc_kfold)
        print(file_name1)
        logger.info(file_name1)
        torch.save(best_model, os.path.join(save_path, file_name1))

        file_name2 = '{}_sub{}_fold{}_acc{:.4}_best_model_step2.pth'.format(model_name, subject, kfold, best_acc_kfold)
        print(file_name2)
        logger.info(file_name2)
        torch.save(model.state_dict(), os.path.join(save_path, file_name2))

        info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info = info + 'The model was saved successfully!'
        print(info)
        logger.info(info + '\n')

        # TEST
        x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test).reshape(-1)
        test_index = torch.LongTensor([i for i, _ in enumerate(y_test)])
        testdataset = TensorDataset(x_test, y_test, test_index)

        # 测试1
        netTest = torch.load(os.path.join(save_path, file_name1))
        model.load_state_dict(netTest, strict=False)
        loss_val, accuracy_val, confusion_val, kappa_val = validate_model(model, testdataset, device, losser,
                                                                          n_calsses=nClass)
        info = 'TEST-step1: Loss:{0:.3}\tacc:{1:.3}\n'.format(loss_val, accuracy_val)
        info += str(confusion_val)
        logger.info(info)
        accuracy_val_list1.append(accuracy_val.to('cpu').numpy())
        kappa_val_list1.append(kappa_val)

        # 测试2
        netTest = torch.load(os.path.join(save_path, file_name2))
        model.load_state_dict(netTest, strict=False)
        loss_val, accuracy_val, confusion_val, kappa_val = validate_model(model, testdataset, device, losser,
                                                                          n_calsses=nClass)
        info = 'TEST-step2: Loss:{0:.3}\tacc:{1:.3}\n'.format(loss_val, accuracy_val)
        info += str(confusion_val)
        logger.info(info)
        accuracy_val_list2.append(accuracy_val.to('cpu').numpy())
        kappa_val_list2.append(kappa_val)

    # after k-fold
    info = f"Avg_eval_Acc : {avg_eval_acc * 100 / kfolds:4f}"
    logger.info(info + '\n')

    logger.info("TESTACC-STAGE1")
    accuracy_val_list = np.array(accuracy_val_list1)
    logger.info(accuracy_val_list.tolist())
    logger.info(np.max(accuracy_val_list))
    logger.info(np.mean(accuracy_val_list))

    kappa_val_list1 = np.array(kappa_val_list1)
    logger.info(kappa_val_list1.tolist())
    logger.info(np.max(kappa_val_list1))
    logger.info(np.mean(kappa_val_list1))

    logger.info("TESTACC-STAGE2")
    accuracy_val_list = np.array(accuracy_val_list2)
    logger.info(accuracy_val_list.tolist())
    logger.info(np.max(accuracy_val_list))
    logger.info(np.mean(accuracy_val_list))
    ret_result = np.max(accuracy_val_list)

    kappa_val_list2 = np.array(kappa_val_list2)
    logger.info(kappa_val_list2.tolist())
    logger.info(np.max(kappa_val_list2))
    logger.info(np.mean(kappa_val_list2))

    return ret_result
