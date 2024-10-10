import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import torch
from covomix.data_module import SpecsDataModule
from covomix.conditional_model import CoVoMixModel

import os

def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          parser_.add_argument("--model_save_dir", type=str,  default="logs")
          parser_.add_argument("--pretrained_model",   type=str, default = "no", help="pretrained model or resume from checkpoint")  


     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     parser = pl.Trainer.add_argparse_args(parser)
     CoVoMixModel.add_argparse_args(
          parser.add_argument_group("CoVoMixModel", description=CoVoMixModel.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     print(args)
     # if args.model_save_dir does not exist
     if not os.path.exists(args.model_save_dir):
          os.makedirs(args.model_save_dir)
     with open (os.path.join(args.model_save_dir, "args.txt"), "w") as f:
          f.write(str(args))
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule
     model = CoVoMixModel(
                         data_module_cls=data_module_cls,
                         medium_file_save_dir = args.model_save_dir, 
                         **{
                              **vars(arg_groups['CoVoMixModel']),
                              **vars(arg_groups['DataModule']), 
                         }
                    )
      
     # Set up logger configuration     
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir=args.model_save_dir, name="tensorboard")
     else:
          logger = WandbLogger(project="covomix", log_model=True, save_dir=args.model_save_dir)
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"{args.model_save_dir}/{logger.version}", save_last=True, filename='{epoch}-last')]
     if args.num_eval_files:
          checkpoint_callback_l2 = ModelCheckpoint(dirpath=f"{args.model_save_dir}/{logger.version}", 
               save_top_k=10, monitor="l2", mode="min", filename='{epoch}-{l2:.2f}')
          callbacks += [checkpoint_callback_l2]

     # Initialize the Trainer and the DataModule
     
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          strategy=DDPPlugin(find_unused_parameters=True), logger=logger,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks
     )

     # Train model
     trainer.fit(model)
     #trainer.validate(model)
