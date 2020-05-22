import logging
import torch
import torch.nn as nn
import time
global ITER
ITER = 0

def do_train(
        cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    start_train(model, optimizer, loss_fn, output_dir, train_loader, epochs, device)

def start_train(model, optimizer, loss_fn, output_dir, train_loader, epochs, device=None):
    if device:
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        model.to(device)
    for epoch in range(epochs):
        for iteration, batch in enumerate(train_loader):
            time_start = time.time()
            optimizer.zero_grad()
            img, target = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            score, feat = model(img)
            loss = loss_fn(score, feat, target)
            loss.backward()
            optimizer.step()
            # compute accuracy
            acc = (score.max(1)[1] == target).float().mean()
            if iteration % 20 == 0:
                time_iteration = time.time()-time_start
                print("loss: {:.3f}, accuracy: {:.1f}%, iteration: {}, time per iteration: {:.3f}".format(loss, acc*100, iteration, time_iteration))
        torch.save(model, output_dir)
        print("epoch: {}-------------------".format(epoch))