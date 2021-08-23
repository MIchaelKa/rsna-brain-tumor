import torch
import time

from utils import format_time

from metrics import AccuracyMeter, AverageMeter

def train_num_iter(model, device, data_loader, criterion, optimizer, max_iter, verbose=True, print_every=10):

    t0 = time.time()

    loss_meter = AverageMeter()
    score_meter = AccuracyMeter()

    model = model.to(device)
    model.train()
    
    data_loader_iter = iter(data_loader)

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

        t_loss_avg = loss_meter.compute_average()
        t_score = score_meter.compute_score()

        if verbose and iter_num % print_every == 0:
            print('[train] iter: {:>4d}, loss = {:.5f}, score = {:.5f}, time: {}'
                .format(iter_num, t_loss_avg, t_score, format_time(time.time() - t0)))


    if verbose:
        print('[train] finished for: {}'.format(format_time(time.time() - t0)))

    return loss_meter, score_meter

