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
from fastreid.evaluation import (DatasetEvaluator, ReidEvaluator,
                                 inference_on_dataset, print_csv_format)

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
    logger = logging.getLogger("fastreid.trainer")
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = H_Trainer.build_model(cfg)

    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    res = H_Trainer.test(cfg, model)
    print_csv_format(res)




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


# train -pc
#python ./tools/train_net.py --config-file ./configs/Hazy_Market1501/bagtricks_R50.yml MODEL.DEVICE "cuda:0" \
# OUTPUT_DIR "/home/pj/fast-reid-master/logs/hazy-market1501/bagtricks_R50/"

# train -server
# export CUDA_VISIBLE_DEVICES= 1,2
# python ./tools/train_net.py --config-file ./configs/Hazy_Market1501/bagtricks_R50.yml --num-gpus 2 SOLVER.IMS_PER_BATCH "32" \
# OUTPUT_DIR "/home/lhf/pj/fast-reid-master/logs/hazy-market1501/bagtricks_R50/"
# python ./tools/train_net.py --config-file ./configs/Hazy_DukeMTMC/bagtricks_R50.yml --num-gpus 2 OUTPUT_DIR "/home/w/pj/fast-reid-master/logs/hazy-dukemtmc/bagtricks_R50/" SOLVER.IMS_PER_BATCH "32"


# evaluation-pc
# ./tools/train_net.py --config-file ./configs/Hazy_Market1501/bagtricks_R50.yml --eval-only \
# MODEL.WEIGHTS /home/pj/fast-reid-master/logs/market1501/bagtricks_R50/model_final.pth MODEL.DEVICE "cuda:0"

# evaluation-server
# ./tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --eval-only \
# MODEL.WEIGHTS /home/lhf/pj/fast-reid-master/logs/market1501/bagtricks_R50/model_final.pth MODEL.DEVICE "cuda:1" \
# OUTPUT_DIR "/home/lhf/pj/fast-reid-master/logs/hazy-market1501/bagtricks_R50/"
# #!/usr/bin/env python
# # encoding: utf-8
# """
# @author:  sherlock
# @contact: sherlockliao01@gmail.com
# """
#
# import logging
# import os
# import sys
#
# sys.path.append('.')
#
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch, Hazytrainer, LoadTrainer
# from fastreid.utils.checkpoint import Checkpointer
# from fastreid.engine import hooks
# from fastreid.evaluation import ReidEvaluator
#
#
# class LTrainer(LoadTrainer):
#     @classmethod
#     def build_evaluator(cls, cfg, num_query, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         return ReidEvaluator(cfg, num_query)
#
#
# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(cfg, args)
#     return cfg
#
#
# def main(args):
#     cfg = setup(args)
#     logger = logging.getLogger("fastreid.trainer")
#     cfg.defrost()
#     cfg.MODEL.BACKBONE.PRETRAIN = False
#     trainer = LTrainer(cfg)
#     trainer.resume_or_load(True)
#     # trainer.test(cfg, trainer.model)
#     return trainer.train()
#
#
# if __name__ == "__main__":
#     args = default_argument_parser()
#     args.add_argument("--info", type=str, default="test", help="information of parameters and losses")
#     args = args.parse_args()
#     print("Command Line Args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )
#
# # train -pc
# #python ./tools/train_net.py --config-file ./configs/Hazy_Market1501/bagtricks_R50.yml MODEL.DEVICE "cuda:0" \
# # OUTPUT_DIR "/home/pj/fast-reid-master/logs/hazy-market1501/bagtricks_R50/"
#
# # train -server
# # export CUDA_VISIBLE_DEVICES= 1,2
# # python ./tools/train_net.py --config-file ./configs/Hazy_Market1501/bagtricks_R50.yml --num-gpus 2 SOLVER.IMS_PER_BATCH "32" \
# # OUTPUT_DIR "/home/lhf/pj/fast-reid-master/logs/hazy-market1501/bagtricks_R50/"
# # python ./tools/train_net.py --config-file ./configs/Hazy_DukeMTMC/bagtricks_R50.yml --num-gpus 2 OUTPUT_DIR "/home/w/pj/fast-reid-master/logs/hazy-dukemtmc/bagtricks_R50/" SOLVER.IMS_PER_BATCH "32"
#
#
# # evaluation-pc
# # ./tools/train_net.py --config-file ./configs/Hazy_Market1501/bagtricks_R50.yml --eval-only \
# # MODEL.WEIGHTS /home/pj/fast-reid-master/logs/market1501/bagtricks_R50/model_final.pth MODEL.DEVICE "cuda:0"
#
# # evaluation-server
# # ./tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --eval-only \
# # MODEL.WEIGHTS /home/lhf/pj/fast-reid-master/logs/market1501/bagtricks_R50/model_final.pth MODEL.DEVICE "cuda:1" \
# # OUTPUT_DIR "/home/lhf/pj/fast-reid-master/logs/hazy-market1501/bagtricks_R50/"