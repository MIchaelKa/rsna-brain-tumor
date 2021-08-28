import torch
import time

from utils import format_time

from metrics import AccuracyMeter, AverageMeter

def validate(model, device, valid_loader, criterion, verbose=True, print_every=10):

    t0 = time.time()

    model.eval()
    
    loss_meter = AverageMeter()
    score_meter = AccuracyMeter()

    with torch.no_grad():
        for iter_num, (x_batch, y_batch) in enumerate(valid_loader):
            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.long)
            
            output = model(x_batch)

            y_batch = y_batch.type_as(output)
            loss = criterion(output.squeeze(1), y_batch)
            
            # Update loss meter
            loss_item = loss.item()
            loss_meter.update(loss_item)

            # Update score meter
            score_meter.update(y_batch, output.squeeze(1))

            if verbose and iter_num % print_every == 0:
                loss_avg = loss_meter.compute_average()
                score = score_meter.compute_score()

                print('[valid] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                    .format(iter_num, loss_avg, score, format_time(time.time() - t0)))
   
    return loss_meter, score_meter

def get_next_valid_iter(valid_iters):
    if len(valid_iters) > 0:
        return valid_iters.pop(0)
    else:
        return -1

def train_num_iter(
    model, device,
    data_loader, valid_loader,
    criterion, optimizer,
    max_iter, valid_iters=[],
    verbose=True, print_every=10):

    t0 = time.time()

    print('[train] started...')

    train_loss_meter = AverageMeter()
    train_score_meter = AccuracyMeter()

    train_loss_history = []
    train_score_history = []

    valid_loss_history = []
    valid_score_history = []

    # model = model.to(device)
    model.train()
    
    data_loader_iter = iter(data_loader)

    valid_iters_copy = valid_iters.copy()
    valid_iter_num = get_next_valid_iter(valid_iters)

    for iter_num in range(0, max_iter):

        t1 = time.time()

        try:
            x_batch, y_batch = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            x_batch, y_batch = next(data_loader_iter)

        x_batch = x_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.long)

        # print(x_batch.shape)

        output = model(x_batch)

        # TODO: will be faster to return float from the dataset?
        y_batch = y_batch.type_as(output)

        # TODO:
        # output = output.squeeze(1)

        # print(output.shape, y_batch.shape)

        loss = criterion(output.squeeze(1), y_batch)
        # loss = criterion(output, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss meter
        loss_item = loss.item()
        train_loss_meter.update(loss_item)

        # Update score meter
        train_score_meter.update(y_batch, output.squeeze(1))

        # if verbose and iter_num % print_every == 0:
        #     t_loss_avg = train_loss_meter.compute_average()
        #     t_score = train_score_meter.compute_score()

        #     print('[train] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
        #         .format(iter_num, t_loss_avg, t_score, format_time(time.time() - t0)))

        
        if iter_num == valid_iter_num:
            valid_iter_num = get_next_valid_iter(valid_iters)

            v_loss_meter, v_score_meter = validate(model, device, valid_loader, criterion, verbose=False, print_every=5)

            # TODO: one more train meters to reset it here?

            t_loss_avg = train_loss_meter.compute_average()
            t_score = train_score_meter.compute_score()
            
            train_loss_history.append(t_loss_avg)
            train_score_history.append(t_score)

            v_loss_avg = v_loss_meter.compute_average()
            v_score = v_score_meter.compute_score()

            valid_loss_history.append(v_loss_avg)
            valid_score_history.append(v_score)

            # TODO: move out and see performance and time
            model.train()

            if verbose:
                print('[train] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                    .format(iter_num, t_loss_avg, t_score, format_time(time.time() - t0)))
                print('[valid] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                    .format(iter_num, v_loss_avg, v_score, format_time(time.time() - t0)))
                print('')

    if verbose:
        print('[train] finished for: {}'.format(format_time(time.time() - t0)))

    train_info = {
        'train_loss_meter' : train_loss_meter,
        'train_score_meter' : train_score_meter,

        'train_loss_history' : train_loss_history,
        'train_score_history' : train_score_history,

        'valid_loss_history' : valid_loss_history,
        'valid_score_history' : valid_score_history,

        'valid_iters' : valid_iters_copy
    }

    return train_info

