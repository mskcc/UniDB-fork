import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class SlabModel(BaseModel):
    def __init__(self, opt):
        super(SlabModel, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        self.model = DataParallel(self.model)
        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()
        else:
            is_weighted = opt['test']['is_weighted']
            loss_type = opt['test']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['test']['weight']
            self.log_dict = OrderedDict()

    def feed_data(self, LQ, GT=None):
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT

    def optimize_parameters(self):

        self.optimizer.zero_grad()

        self.output = self.model(self.condition)
        
        loss = self.loss_fn(self.output, self.state_0)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def test(self, sde=None, save_states=False):
        # TODO 
        return None
        
    def check_parameters(self, val_data=True):

        self.model.eval()

        self.output = self.model(self.condition)

        loss = self.loss_fn(self.output, self.state_0)

        if val_data == True:
            # set log
            self.log_dict["val_loss"] = loss.item()

        self.model.train()

        return loss.item()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
   
    def save_best(self, iter_label):
        self.save_network(self.model, "best_G", 'PSNR')
        self.save_network(self.ema.ema_model, "best_EMA", 'PSNR')

    def save_best_L1(self, iter_label):
        self.save_network(self.model, "best_G", 'L1')
        self.save_network(self.ema.ema_model, "best_EMA",'L1')
