#!/usr/bin/env python
"""Training script."""
import datetime
import logging
import pprint
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import chino.configurator as cc
import chino.timer as ct
from chino.setup_logging import setup_logging
from configs import cfg
from point_net import PointNet
from modelnet.modelnet import ModelNetCls, PCAugmentation, collate_fn

logger = logging.getLogger(__name__)


def parse_args():
    """Argument parser."""
    parser = cc.cfg_parser()
    parser.add_argument('--cfg', dest='cfg_file', type=str, default=None,
                        help='YAML config file to parse from.')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def train_model():
    """Main function for training classification model."""
    dataset = ModelNetCls(cfg.DATA_PATH,
                          modelnet40=(cfg.DATASET=='modelnet40'),
                          train=True, transform=PCAugmentation(),
                          num_points=cfg.NUM_POINTS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    point_net = PointNet(3, 40).to(device)
    for m in point_net.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = cfg.BN_MOMENTUM
    if cfg.OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(point_net.parameters(), lr=cfg.LR,
                                    momentum=cfg.MOMENTUM,
                                    weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(point_net.parameters(), lr=cfg.LR,
                                     weight_decay=cfg.WEIGHT_DECAY)
    else:
        raise ValueError('Unknown optimizer: {}'.format(cfg.OPTIMIZER))
    for e in range(cfg.EPOCHS):
        ct.tic('epoch')
        point_net.train()
        logger.info('Training on epoch %d/%d', e+1, cfg.EPOCHS)
        loader = torch.utils.data.DataLoader(
            dataset, cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
            shuffle=True, collate_fn=collate_fn, pin_memory=True,
            drop_last=True,
        )
        ct.tic('batch')
        for batch_idx, (data, labels) in enumerate(loader):
            optimizer.zero_grad()
            data, labels = data.to(device), labels.to(device)
            out = point_net(data)
            _, predicted = out.max(dim=-1)
            loss = F.cross_entropy(out, labels)
            train_accuracy = (predicted==labels).sum().item() / len(labels)
            if ct.toc('batch') > 5:
                logger.info(
                    '%d/%d for epoch %d, '
                    'Cls loss: %.3f, '
                    'train acc: %.3f',
                    batch_idx*cfg.BATCH_SIZE, len(dataset), e+1,
                    loss.item(), train_accuracy
                )
                ct.tic('batch')
            loss.backward()
            optimizer.step()
        if cfg.SNAPSHOT > 0 and ((e + 1) % cfg.SNAPSHOT == 0):
            filename = os.path.join(cfg.OUTPUT_PATH, '{}.{}-{}.pth'.format(
                'PointNet2Cls', cfg.DATASET, e+1
            ))
            logger.info('Saving model to %s', filename)
            torch.save(point_net.state_dict(), filename)
        #  if cfg.TEST_INTERVAL > 0 and ((e+1) % cfg.TEST_INTERVAL == 0):
        #      logger.info('Running test for epoch %d/%d', e+1, cfg.EPOCHS)
        #      ins_acc, cls_acc = test_model(point_net)
        #      logger.info('Instance accuracy: %.3f, class accuracy: %.3f',
        #                  ins_acc, cls_acc)
        # update learning rate
        if (cfg.STEPSIZE > 0) and ((e+1) % cfg.STEPSIZE == 0):
            cfg.LR = max(cfg.LR * cfg.GAMMA, cfg.MIN_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.LR
            logger.info('Learning rate set to %g', cfg.LR)
        # update bn momentum
        if (cfg.BN_STEPSIZE > 0) and ((e+1) % cfg.BN_STEPSIZE == 0):
            cfg.BN_MOMENTUM = max(cfg.BN_MOMENTUM*cfg.BN_GAMMA,
                                  cfg.BN_MIN_MOMENTUM)
            for m in point_net.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.momentum = cfg.BN_MOMENTUM
            logger.info('BatchNorm momentum set to %g', cfg.BN_MOMENTUM)
        logger.info('Elapsed time for epoch %d: %.3fs', e+1, ct.toc('epoch'))
    if cfg.SNAPSHOT > 0:
        filename = os.path.join(cfg.OUTPUT_PATH, '{}.{}.pth'.format(
            'PointNet2Cls', cfg.OUTPUT_PATH
        ))
        logger.info('Saving final model to %s', filename)
        torch.save(point_net.state_dict(), filename)


def main():
    """Main entry."""
    args = parse_args()
    cc.merge_from_parser_args(args)
    if args.cfg_file is not None:
        cc.merge_from_yml(args.cfg_file)
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)
    log_name = os.path.join(
        cfg.OUTPUT_PATH,
        '{:s}.{:%Y-%m-%d_%H-%M-%S}.{:s}.log'.format(
            cfg.DATASET,
            datetime.datetime.now(),
            'train_test' if cfg.TEST_INTERVAL > 0 else 'train'
        )
    )
    setup_logging(log_name)
    logger.info('Configs:')
    logger.info(pprint.pformat(cfg))
    if cfg.RNG_SEED >= 0:
        np.random.randn(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
    train_model()


if __name__ == "__main__":
    main()
