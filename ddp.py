import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *

from train_oneshot import train
from argparse import ArgumentParser


torch.manual_seed(1)
random.seed(1)


class OneShotCLIP(pl.LightningModule):
  def __init__(self):
    self.

  
  def training_batch(self):

  def configure_optimizers(self):
    
    opt = optim.Adam([self.var_img], lr = self.lr, betas = (self.beta_1, self.beta_2))
    return opt
  
if __name__ == '__main__':
  device = "cuda"
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--iter", type=int, default=2001)
  parser.add_argument("--save_freq", type=int, default=1000)
  parser.add_argument("--img_freq", type=int, default=100)
  parser.add_argument("--highp", type=int, default=1)
  parser.add_argument("--ref_freq", type=int, default=4)
  parser.add_argument("--feat_ind", type=int, default=3)
  parser.add_argument("--batch", type=int, default=2)
  parser.add_argument("--n_sample", type=int, default=4)
  parser.add_argument("--size", type=int, default=1024)

  parser.add_argument("--r1", type=float, default=10)

  parser.add_argument("--d_reg_every", type=int, default=16)
  parser.add_argument("--g_reg_every", type=int, default=4)
  parser.add_argument("--mixing", type=float, default=0.9)

  parser.add_argument("--ckpt", type=str, default=None)

  parser.add_argument("--exp", type=str, default=None, required=True)
  parser.add_argument("--lr", type=float, default=0.002)
  parser.add_argument("--f_lr", type=float, default=0.01)
  parser.add_argument("--channel_multiplier", type=int, default=2)

  # DDP 관련
  parser.add_argument('-gpus', '--gpus', type=int, default=1)
  parser.add_argument('-mode', '--strategy', type=str, default=None)
  parser.add_argument('-nnode', '--nodes', type=int, default=1)
  parser.add_argument("--local_rank", type=int, default=0)
  parser.add_argument('-precision', '--precision', type=int, default= 32)
  parser.add_argument('-num_workers', '--num_workers', type=int, default=8)
  
  parser.add_argument("--skip_init",action='store_true')
  parser.add_argument("--init_iter", type=int, default=1001)
  parser.add_argument("--lambda_optclip", type=float, default=1)
  parser.add_argument("--lambda_optl2", type=float, default=0.01)
  parser.add_argument("--lambda_optrec", type=float, default=1)
  parser.add_argument("--lambda_patch", type=float, default=1)
  parser.add_argument("--lambda_const", type=float, default=10)
  parser.add_argument("--crop_size", type=int, default=128)
  parser.add_argument("--num_crop", type=int, default=16)
  parser.add_argument("--cars", action="store_true")
  parser.add_argument("--nce_allbatch", action="store_true")
  parser.add_argument("--tau", type=float, default=1.0)
  args = parser.parse_args()


  tqdm_cb = TQDMProgressBar(refresh_rate=10, process_position=1)
  trainer = Trainer(max_epochs=args.epochs, gpus=args.gpus,
                    strategy=args.strategy, num_nodes = args.nodes,
                    callbacks=[tqdm_cb], enable_checkpointing=True, precision=args.precision)

  trainer.fit(model, cifar10dm)