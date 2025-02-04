import os
import shutil
import argparse
from easydict import EasyDict
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm

from utils.misc import BlackHole, get_logger, get_new_log_dir, load_config, seed_all, Counter
from utils.train import get_optimizer, get_scheduler, log_losses
from .datasets import PSRDataset, PairBatchSampler
from .models import PSRNetwork
from .utils import report_correlations


parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--logdir', type=str, default='./logs/psr')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--overwrite', action='store_true', default=False)
args = parser.parse_args()

# Load configs
config, config_name = load_config(args.config)
seed_all(config.train.seed)

# Logging
if args.debug:
    logger = get_logger(config_name, None)
    writer = BlackHole()
else:
    if args.resume is not None and args.overwrite:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
        shutil.copytree('./modules', os.path.join(log_dir, 'modules'))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

# Dataloaders
logger.info('Loading datasets...')
train_set = PSRDataset(os.path.join(config.data.root, 'train'))
val_set = PSRDataset(os.path.join(config.data.root, 'test'))
train_loader = DataLoader(train_set, batch_sampler=PairBatchSampler(train_set.target_to_indices, config.data.train_batch_size))
val_loader = DataLoader(val_set, batch_size=config.data.val_batch_size)
logger.info('Train: %d | Validation: %d' % (len(train_set), len(val_set)))

# Model
logger.info('Building model...')
model = PSRNetwork(**config.model).to(args.device)
global_step = Counter()

# Optimizer
optimizer = get_optimizer(config.train.optimizer, model)
scheduler = get_scheduler(config.train.scheduler, optimizer)

# Resume
it_first = 1
if args.resume is not None:
    logger.info('Resuming from checkpoint: %s' % args.resume)
    ckpt = torch.load(args.resume, map_location=args.device)
    it_first = ckpt['iteration']
    model.load_state_dict(ckpt['model'])
    logger.info('Resuming optimizer and scheduler states...')
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])

def train(it):
    model.train()
    for i, batch in enumerate(tqdm(train_loader, desc='Train', position=0, leave=True)):
        batch = batch.to(args.device)
        optimizer.zero_grad()
        output = model(batch)   # (Pair*2, )
        target = batch.gdt_ts
        loss_single = F.huber_loss(output, target)

        output_pair = output.reshape(-1, 2) # (Pair, 2)
        target_pair = target.reshape(-1, 2) # (Pair, 2)
        loss_pair = F.huber_loss(output_pair[:,0] - output_pair[:,1], target_pair[:,0] - target_pair[:,1])

        loss = loss_single + loss_pair
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        
        log_others = {
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
        }
        log_losses(EasyDict({'overall': loss, 'single': loss_single, 'pair': loss_pair}), global_step.step(), 'train', logger=BlackHole(), writer=writer, others=log_others)


def validate(it):
    y_true, y_pred, targets, decoys = [], [], [], []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(val_loader)):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            output = model(batch)   # (G, )

            y_pred.extend([v.item() for v in output])
            y_true.extend([v.item() for v in batch.gdt_ts])
            targets.extend(batch.target_id)
            decoys.extend(batch.decoy_id)

    test_df = pd.DataFrame(
        np.array([targets, decoys, y_true, y_pred]).T,
        columns=['target', 'decoy', 'true', 'pred'],
    )
    print(test_df)
    corrs = report_correlations(test_df, logger, writer, it)
    return np.mean(corrs['all_pearson']) + np.mean(corrs['per_target_pearson'])


try:
    for it in range(it_first, config.train.max_epochs+1):
        train(it)
        if it % config.train.val_freq == 0:
            avg_val_loss = validate(it)
            scheduler.step()
            if not args.debug:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
except KeyboardInterrupt:
    logger.info('Terminating...')
    
