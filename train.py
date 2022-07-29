from logging import Logger
from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import show_val_samples
import utils
from config import config


def train(
    train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, save_dir
):
    # training loop
    lr = config.start_lr
    logdir = "./results/tensorboard/net"
    logger = utils.make_logger(save_dir+"/logs.txt")
    logger.info('Starting')
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)
    best_stats = {}
    history = {}  # collects metrics at the end of each epoch
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {"loss": [], "val_loss": []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics["val_" + k] = []

        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        # training
        model.train()
        for i, (x, y) in enumerate(pbar):
            # configure custom scheduler in config
            
            if config.use_custom_lr_scheduler:
                lr = utils.learning_rate(optimizer, lr, train_dataloader.dataset.n_samples,
                i, warmup_iter=config.warm_up_epochs, power=0.9)
            y = torch.squeeze(y)
            optimizer.zero_grad()  # zero out gradients
            y_hat = torch.squeeze(model(x))  # forward pass

            loss = loss_fn(y_hat, y)

            if  config.GRAD_ACCUM > 1:
                loss = loss/config.GRAD_ACCUM
                #loss = loss
            
            loss.backward()  # backward pass
            if (i + 1) % config.GRAD_ACCUM == 0:
                optimizer.step()  # optimize weights
                optimizer.zero_grad()

            # log partial metrics
            metrics["loss"].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix(
                {k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0}
            )

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y = torch.squeeze(y)
                y_hat = model(x)  # forward pass
                y_hat = torch.squeeze(y_hat)
                loss = loss_fn(y_hat, y)
                # log partial metrics
                metrics["val_loss"].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics["val_" + k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        best_stats = utils.save_if_best_model(model, save_dir, epoch, history, best_stats, logger)
        for k, v in history[epoch].items():
            writer.add_scalar(k, v, epoch)
        print(
            " ".join(
                [
                    "\t- " + str(k) + " = " + str(v) + "\n "
                    for (k, v) in history[epoch].items()
                ]
            )
        )
        #show_val_samples(
        #    x.detach().cpu().numpy(),
        #    y.detach().cpu().numpy(),
        #    y_hat.detach().cpu().numpy(),
        #)

    print("Finished Training")


    
    # plot loss curves
    plt.plot([v["loss"] for k, v in history.items()], label="Training Loss")
    plt.plot([v["val_loss"] for k, v in history.items()], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
