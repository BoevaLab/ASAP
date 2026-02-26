from datetime import datetime
from typing import Callable, Union, Tuple, Dict
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from random import randint

from asap.utils.optim import configure_adamw, make_warmupCAWR
from asap.utils.metrics import compute_metrics
from asap.utils.logger import Logger


def prepend_to_keys(d: dict, prepend: str) -> Dict:
    return {prepend + key: d[key] for key in d}

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 filename: str = None,
                 criterion: Union[Callable, None] = None,
                 unmap_criterion: Union[Callable, bool, None] = None,
                 batch_size: int = None,
                 logger: Logger = None,
                 n_gpus: int = None,
                 ):
        self.filename = filename
        self.model = model
        self.criterion = criterion 
        self.unmap_criterion = nn.MSELoss() if unmap_criterion is True else None
        self.train_unmap = not self.unmap_criterion is False

        self.logger: Logger = logger
        self.logspace = True 
        self.nr_tracks = 1
        self.nr_devices = n_gpus
        self.batch_size = batch_size
        if self.nr_devices > 1: 
            self.ddp_enabled = True
            self.device = 'cuda'
            torch.backends.cudnn.enabled = False
        elif self.nr_devices == 1:
            self.ddp_enabled = False
            self.device = 'cuda'
            self.model.to(self.device)
        else:
            self.ddp_enabled = False
            self.device = 'cpu'
            self.model.to(self.device)

    def fit(self, train_dset, val_dset, nr_epochs, learning_rate):
        print(f'Training {self.filename}...')
        if self.nr_devices > 1:
            port = 10000 + randint(0,2355)
            mp.spawn(
                _ddp_and_fit,
                args=(
                    self.model,
                    train_dset,
                    val_dset,
                    self.batch_size // self.nr_devices, # batch per GPU
                    nr_epochs,
                    learning_rate,
                    self.criterion,
                    self.unmap_criterion,
                    self.logger,
                    self.filename,
                    self.nr_devices,
                    port
                ),
                nprocs=self.nr_devices
            )
        else:
            self.model.to(self.device)
            train_gen = make_dataloader(
                ddp_enabled=False,
                dataset=train_dset,
                batch_size=self.batch_size,
                is_train=True
            )
            val_gen = make_dataloader(
                ddp_enabled=False,
                dataset=val_dset,
                batch_size=self.batch_size,
                is_train=False
            )
            _fit(    
                self.device,
                self.model,
                train_gen,
                val_gen,
                nr_epochs,
                learning_rate,
                self.criterion,
                self.unmap_criterion,
                self.logger,
                self.filename,
                ddp_enabled=False
            )

    def predict(self, gen):
        self.model.to(self.device)
        predictions, true = _predict(self.model, gen, self.device, False)

        return torch.cat(predictions), torch.cat(true)

    def evaluate(self, test_gen) -> dict:
        print(f'Evaluating {self.filename}...')
        result_metrics = self.predict_and_evaluate(test_gen)
        print(result_metrics)
        return result_metrics

    def predict_and_evaluate(self, gen, metrics_for_track=None, no_eval=False) -> Tuple[np.ndarray, np.ndarray, dict]:
        self.model.eval()

        try:
            gen.dataset.margin_size
        except AttributeError:
            print("Generator has no margin size -- assuming full prediction.")

        predictions, true = self.predict(gen)
        predictions = predictions.reshape((-1, self.nr_tracks)).detach().cpu().numpy()
        true = true.reshape((-1, self.nr_tracks)).detach().cpu().numpy()
        
        if no_eval:
            return true, predictions, None

        metric_results = {}
        if metrics_for_track is None:
            # compute metrics for all tracks
            metrics_for_track = range(self.nr_tracks)
            metrics_ = compute_metrics(
                    predictions.flatten(),
                    true.flatten(),
                    logspace_input=self.logspace
                )
            metric_results.update(metrics_)
        else:
            for track in metrics_for_track:
                metrics_ = compute_metrics(
                    predictions[:, track],
                    true[:, track],
                    logspace_input=self.logspace
                )
                metric_results.update(prepend_to_keys(metrics_, f"_{track}_"))
        return true, predictions, metric_results

    def predict_robust_batch(self, test_gen, bin_size=4, nr_samples_for_var=17,
                             window=512, margin=768) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #              |---m---|-----w-----|---m---| # input size
        # |-----p-----|---------vc---------|-----p-----| # valid center + padding, vc=input_size//2
        #          |-----p-----|---------vc---------|-----p-----|
        # |--------------------------t'-------------------------| # total size = 2 (p + vc) - w
        # |---------m'---------|-----w-----|---------m'---------| # margin_size' = (t' - w) // 2
        ys = None
        ys_pred = None
        variance = np.empty(shape=0)
        variance_per_bin = np.empty(shape=(0, window // bin_size))
        gen_tracks = 1
        model_out = window + 2 * margin

        assert (0 == margin % (nr_samples_for_var - 1))
        shift_bp_step = window //  (nr_samples_for_var - 1) # -1 to include endpoint

        idx = np.array([np.arange(window + 2 * margin) + shift_bp_step * i for i in range(nr_samples_for_var)])
        idx_pred = (np.array([
            model_out // 2 # to center
            + np.arange(-window//2, window//2, step=bin_size) # range symmetric around center
            + shift_bp_step * (-i - 1) for i in range(-nr_samples_for_var//2, nr_samples_for_var//2) # shift to align
            ]) // bin_size) # divide by bin_size

        self.model.eval()
        with torch.no_grad():
            # Loop over data
            for X_i, _, y_i in tqdm(test_gen):
                X_i = X_i.to(self.device)

                if ys is None:
                    gen_tracks = 1  # only for single task, adjust code if model has more tracks
                    ys = np.empty(shape=(0, window // bin_size, gen_tracks))
                    ys_pred = np.empty(shape=(0, nr_samples_for_var, window // bin_size, gen_tracks))

                batch_size = X_i.shape[0]
                y_i = y_i.reshape((batch_size, y_i.shape[1], gen_tracks))
                assert len(X_i.shape) == 3 # (batch, seq, base)
                X_i = X_i[:, idx]
                X_i = X_i.reshape((-1, window + 2 * margin, 4))
                # get shifted predictions
                try:
                    shifts_y_pred = self.model(X_i).cpu().detach().numpy()
                except:
                    print(f'Failed prediction x={X_i.shape}, y={y_i.shape}')
                    continue
                y_i = y_i.numpy()

                shifts_y_pred = shifts_y_pred.reshape((batch_size, nr_samples_for_var, (window + 2* margin)//bin_size, self.nr_tracks)) # (batch, samples, bins, tracks)
                shifts_y_pred = np.stack([shifts_y_pred[:, i, idx_pred[i]] for i in range(nr_samples_for_var)], axis=1) # (batch, samples, bins, tracks), bins aligned, 1024//4
                ys_pred = np.append(ys_pred, shifts_y_pred, axis=0) # collect predictions, mean over samples

                # collect ground truth
                ys = np.append(ys, y_i, axis=0)

                bin_scores = shifts_y_pred.std(axis=1) / shifts_y_pred.mean(axis=1) # std, mean over samples
                sample_score = bin_scores.mean(axis=1).reshape(-1) # mean over bins, flatten
                bin_scores = np.swapaxes(bin_scores, 1, 2) # (batch, tracks, bins)
                bin_scores = bin_scores.reshape(-1, window // bin_size) # flatten, (batch * tracks, bins)

                variance = np.append(variance, sample_score, axis=0)
                variance_per_bin = np.append(variance_per_bin, bin_scores, axis=0)

        return ys.squeeze(), ys_pred.squeeze(), np.array(variance), np.array(variance_per_bin)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # For loading a DDP model in non-DDP setting
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
            self.model.load_state_dict(state_dict)

def make_dataloader(ddp_enabled, dataset, batch_size: int, is_train: bool, num_workers: int = 0, pin_memory: bool = True):
    # if using DDP, use DistributedSampler
    if ddp_enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=is_train
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=False, # shuffling handled by sampler
            num_workers=num_workers,
            sampler=sampler
        )
    else:
        # if not using DDP, use regular DataLoader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=is_train,
            num_workers=num_workers
        )


# Need to be picklable so seperated out from the trainer
def setup_ddp(rank, world_size, model, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    return model


def _ddp_and_fit(
        rank,
        model,
        train_dset,
        val_dset,
        batch_size,
        nr_epochs,
        learning_rate,
        criterion,
        unmap_criterion,
        logger,
        filename,
        world_size,
        port=12355
    ):
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = setup_ddp(rank, world_size, model, port)
    train_gen = make_dataloader(
        ddp_enabled=True,
        dataset=train_dset,
        batch_size=batch_size,
        is_train=True
    )
    val_gen = make_dataloader(
        ddp_enabled=True,
        dataset=val_dset,
        batch_size=batch_size,
        is_train=False
    )
    _fit(
        rank=rank,
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        nr_epochs=nr_epochs,
        learning_rate=learning_rate,
        criterion=criterion,
        unmap_criterion=unmap_criterion,
        logger=logger,
        filename=filename,
        ddp_enabled=True
    )
    dist.destroy_process_group()


def _fit(
        rank,
        model,
        train_gen,
        val_gen,
        nr_epochs,
        learning_rate,
        criterion,
        unmap_criterion,
        logger: Logger,
        filename: str,
        ddp_enabled: bool
    ):
    optimizer = configure_adamw(model, lr=learning_rate)
    scheduler: torch.optim.lr_scheduler.SequentialLR = make_warmupCAWR(
        optimizer=optimizer,
        warmup_steps=int(len(train_gen) * 0.25),
        n_per_epoch=len(train_gen),
        gamma=0.9
    )
    early_stopping_after_no_improvement = 5 # Set to 0 for no early stopping
    no_improvement_for = 0
    best_val_score = -1

    for epoch in range(nr_epochs):
        if ddp_enabled:
            train_gen.sampler.set_epoch(epoch)
        train_log_payload = _train_epoch(
            rank,
            model,
            train_gen,
            optimizer,
            scheduler,
            criterion,
            unmap_criterion)

        if train_log_payload is not None and (not ddp_enabled or rank == 0):
            logger.log(train_log_payload, step=epoch)

        val_res = _predict(model, val_gen, rank, ddp_enabled=ddp_enabled)

        # For synchronous loop breaking
        stop_early = torch.zeros(1).to(rank)

        if not ddp_enabled or rank == 0:
            logger.log({'lr': scheduler.get_last_lr()[0]})
            predictions, true = val_res
            predictions, true = torch.cat(predictions).cpu(), torch.cat(true).cpu()
            predictions, true = predictions[..., 0].flatten().numpy(), true[..., 0].flatten().numpy()
            val_log_payload = compute_metrics(
                predictions,
                true,
                logspace_input=val_gen.dataset.logspace
            )
            val_log_payload = prepend_to_keys(val_log_payload, 'val/')
            logger.log(val_log_payload, step=epoch)
            print(f'Epoch {epoch} - {datetime.now()}')
            if train_log_payload is not None:
                print(f'\tTrain loss: {train_log_payload["train/loss"]}')
            print(f'\tVal pearson r: {val_log_payload["val/pearson_r"]}')
            print('-----------------------------------------')
            if val_log_payload['val/pearson_r'] > best_val_score:
                best_val_score = val_log_payload['val/pearson_r']
                logger.save_model(model, filename)
                 # handle early stopping
                no_improvement_for = 0
            else:
                no_improvement_for += 1
                if (epoch != nr_epochs -1) and early_stopping_after_no_improvement and no_improvement_for >= early_stopping_after_no_improvement:
                    stop_early += 1

        if ddp_enabled:
            dist.all_reduce(stop_early)
        if stop_early == 1:
            if ddp_enabled:
                dist.barrier() # sync
            if not ddp_enabled or rank == 0:
                print(f"No improvement for {no_improvement_for} epochs. Terminating training early.")
            break

        if ddp_enabled:
            dist.barrier() # sync
    print('Completed training!')


def _train_epoch(rank, model, train_gen, optimizer, scheduler, criterion, unmap_criterion):
    model.train()

    train_unmap = unmap_criterion is not None

    if rank == 0:
        # pbar if on rank 0
        train_gen = tqdm(train_gen)

    for X_i, m_i, y_i in train_gen:
        X_i = X_i.to(rank)
        m_i = m_i.to(rank)
        y_i = y_i.to(rank)
        optimizer.zero_grad()
        if train_unmap:
            output, output_m_i = model(X_i, return_unmap=True)
            # trim m_i in case of unpadded conv in stem
            m_len = output_m_i.shape[1]
            unmap_loss = unmap_criterion(output_m_i, m_i[:, :m_len])

            base_loss = criterion(output, y_i)
            loss = base_loss + unmap_loss
        else:
            output = model(X_i)
            loss = criterion(output, y_i)

        loss.backward()
        optimizer.step()
        scheduler.step()


def compare_tensor(tsr, rank, prefix='', mode='bool', ref=None):
    # Leaving this here in case any debugging of DDP is needed
    if tsr is None:
        dist.barrier()
        if rank == 0:
            print(f"{prefix}:\n", "found None") 
        return
    res = [torch.zeros_like(tsr) for _ in range(dist.get_world_size())]
    dist.all_gather(res, tsr)
    if rank == 0:
        if ref is None:
            ref = res[0].detach().cpu().numpy()
        if mode == 'bool':
            res = [np.allclose(res[rk].detach().cpu().numpy(), ref) for rk in range(dist.get_world_size())]
        elif mode=='mse':
            res = [np.sum((res[rk].detach().cpu().numpy() - ref)**2) for rk in range(dist.get_world_size())]
        print(f"{prefix}:\n", res)


def _predict(model, gen, rank, ddp_enabled):
    model.eval()  # Set the model to evaluation mode
    try:
        margin_size = gen.dataset.margin_size
        bin_size = gen.dataset.bin_size
        trim = margin_size//bin_size
    except AttributeError:
        margin_size = None

    predictions = []
    true = []

    for i, (X_i, _, y_i) in enumerate(gen):
        X_i = X_i.to(rank)
        y_i = y_i.to(rank)
        if i == 2040:
            print(X_i[0,1005:-1005,:])

        with torch.no_grad():
            p_i = model(X_i)

        if margin_size is not None:
            y_i.flatten()
            p_i = p_i[..., trim:-trim, :]
        
        y_i, p_i = y_i.contiguous(), p_i.contiguous()
        if ddp_enabled:
            all_predictions = [torch.zeros_like(y_i) for _ in range(dist.get_world_size())]
            all_true = [torch.zeros_like(y_i) for _ in range(dist.get_world_size())]
            dist.all_gather(all_predictions, p_i)
            dist.all_gather(all_true, y_i)

            if rank == 0:
                predictions.extend(all_predictions)
                true.extend(all_true)
        else:
            predictions.append(p_i)
            true.append(y_i)

    print(i)
    return predictions, true
