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
            t_loss = loss.item()
            loss_meter.update(t_loss)

            # Update score meter
            score_meter.update(y_batch, output.squeeze(1))

            if verbose and iter_num % print_every == 0:
                t_loss_avg = loss_meter.compute_average()
                t_score = score_meter.compute_score()

                print('[valid] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                    .format(iter_num, t_loss_avg, t_score, format_time(time.time() - t0)))
   
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

    loss_meter = AverageMeter()
    score_meter = AccuracyMeter()

    valid_loss_history = []
    valid_score_history = []

    # model = model.to(device)
    model.train()
    
    data_loader_iter = iter(data_loader)

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

        output = model(x_batch)

        # TODO: will be faster to return float from the dataset?
        y_batch = y_batch.type_as(output)

        loss = criterion(output.squeeze(1), y_batch)
        # loss = criterion(output, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss meter
        t_loss = loss.item()
        loss_meter.update(t_loss)

        # Update score meter
        score_meter.update(y_batch, output.squeeze(1))

        if verbose and iter_num % print_every == 0:
            t_loss_avg = loss_meter.compute_average()
            t_score = score_meter.compute_score()

            print('[train] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                .format(iter_num, t_loss_avg, t_score, format_time(time.time() - t0)))

        
        if iter_num == valid_iter_num:
            v_loss_meter, v_score_meter = validate(model, device, valid_loader, criterion, verbose=False, print_every=5)
            valid_iter_num = get_next_valid_iter(valid_iters)

            model.train()

            v_loss_avg = loss_meter.compute_average()
            v_score = score_meter.compute_score()

            valid_loss_history.append(v_loss_avg)
            valid_score_history.append(v_score)

            if verbose:
                print('[valid] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                    .format(iter_num, v_loss_avg, v_score, format_time(time.time() - t0)))

    if verbose:
        print('[train] finished for: {}'.format(format_time(time.time() - t0)))

    train_info = {
        'train_loss_meter' : loss_meter,
        'train_score_meter' : score_meter,
        'valid_loss_history' : valid_loss_history,
        'valid_score_history' : valid_score_history,
    }

    return train_info

