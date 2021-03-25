#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch, Hazytrainer
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import hooks
from fastreid.evaluation import ReidEvaluator


class H_Trainer(Hazytrainer):
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ReidEvaluator(cfg, num_query)

class BaseTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ReidEvaluator(cfg, num_query)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)


    if args.eval_only:
        logger = logging.getLogger("fastreid.trainer")
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = H_Trainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(model):
            prebn_cfg = cfg.clone()
            prebn_cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
            prebn_cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN
            logger.info("Prepare precise BN dataset")
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                model,
                # Build a new data loader to not affect training
                H_Trainer.build_train_loader(prebn_cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ).update_stats()
        res = H_Trainer.test(cfg, model)
        return res

    trainer = H_Trainer(cfg)
    args.resume = True
    trainer.resume_or_load(resume=args.resume)
    # trainer.test(cfg, trainer.model)
    # Checkpointer().load(cfg.MODEL.WEIGHTS)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument("--info", type=str, default="Test", help="information of parameters and losses")
    args = args.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

