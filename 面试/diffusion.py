# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNIST autoencoder example.

To run: python autoencoder.py --trainer.max_epochs=50

"""

from os import path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from lightning.pytorch import LightningDataModule, LightningModule, Trainer, callbacks
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms
    from torchvision.utils import save_image

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import os

def get_dir_path():
    out = path.join(path.dirname(__file__),"Datasets")
    return out

class ImageSampler(callbacks.Callback):
    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        value_range: Optional[tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            value_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.value_range = value_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def _to_grid(self, images):
        return torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.value_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not _TORCHVISION_AVAILABLE:
            return

        images, _ = next(iter(DataLoader(trainer.datamodule.mnist_val, batch_size=self.num_samples)))
        images_flattened = images.view(images.size(0), -1)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images_generated = pl_module(images_flattened.to(pl_module.device))
            pl_module.train()

        if trainer.current_epoch == 0:
            save_image(self._to_grid(images), f"grid_ori_{trainer.current_epoch}.png")
        save_image(self._to_grid(images_generated.reshape(images.shape)), f"grid_generated_{trainer.current_epoch}.png")

class ResNet(nn.Module):
    def __init__(self,net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net=net

    def forward(self,input_data):
        return input_data+self.net(input_data)

class LinearNet(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,layout_num,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
        )
        mid_net_list=[]
        for i in range(layout_num):
            # mid_net_list.append(nn.Linear(hidden_dim,hidden_dim))
            # mid_net_list.append(nn.BatchNorm1d(hidden_dim))
            # mid_net_list.append(nn.LeakyReLU())
            mid_net_list.append(ResNet(
                nn.Sequential(
                    nn.Linear(hidden_dim,hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                )
            ))
            # mid_net_list.append(nn.Dropout2d())
        self.mid_net=nn.Sequential(
            *mid_net_list
        )
        self.post_net=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
        
        

    def forward(self,input_data):
        hidden_data=self.pre_net(input_data)
        hidden_data3=self.mid_net(hidden_data)
        out=self.post_net(hidden_data3)
        return out
    
class Diffusion(nn.Module):
    def __init__(self,input_dim,hidden_dim,group_num,layout_num,step_num,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.LeakyReLU(),
        )

        self.net=LinearNet(hidden_dim,input_dim,hidden_dim,layout_num)

        self.pos_params=nn.Parameter(torch.rand(step_num,hidden_dim))
        
        

    def forward(self,input_data,step):
        hidden_data=self.pre_net(input_data)

        pos_data=self.pos_params[step].view(input_data.shape[0],-1)
        hidden_data2=hidden_data+pos_data

        out=self.net(hidden_data2)
        return out



class Encoder(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,layout_num,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_net=LinearNet(input_dim,hidden_dim,hidden_dim,layout_num-1)
        self.mean_net=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
        self.logvar_net=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
        
        

    def forward(self,input_data):
        input_data=input_data.reshape(input_data.shape[0],-1)
        hidden_data=self.pre_net(input_data)
        mean=self.mean_net(hidden_data)
        logvar=self.logvar_net(hidden_data)
        return mean,logvar
  

class VAE(nn.Module):
    def __init__(self,H,W,C,hidden_dim,layout_num,embed_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h=H
        self.w=W
        self.c=C

        input_dim=H*W*C
        self.encoder=Encoder(input_dim,embed_dim,hidden_dim,layout_num)
        self.decoder=LinearNet(embed_dim,input_dim,hidden_dim,layout_num)
        

    def forward(self,input_data):
        input_shape=input_data.shape
        mean,logvar=self.encoder(input_data)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        # z=mean

        recon_data=self.decoder(z)
        recon_data=recon_data.reshape(input_shape)

        return recon_data,mean,logvar



class DiffusionLightning(LightningModule):
   

    def __init__(self, net,latent_dim,
                 step_num=1000,beta_start=1e-4,beta_end=0.02):
        super().__init__()
        self.step_num=step_num
        self.beta_start=beta_start
        self.beta_end=beta_end
        self.latent_dim=latent_dim
        self.net=net
        self.init_diffusion_params()
    
    def init_diffusion_params(self):

        self.beta=torch.linspace(self.beta_start,self.beta_end,self.step_num).cuda()

        self.alpha=1-self.beta

        self.hat_alpha=torch.cumprod(self.alpha,dim=-1)



    def add_noise(self,input_data):
        B=input_data.shape[0]
        noise_data=torch.randn_like(input_data)
        step_data=torch.randint(0,self.step_num,(B,))
        hat_alpha=self.hat_alpha[step_data].view(B,1)
        img_corr=torch.sqrt(hat_alpha)
        noise_corr=1-hat_alpha

        out_img=img_corr*input_data+noise_corr*noise_data
        return out_img,step_data,noise_data



    def forward(self, noise_img,step):
        return self.net(noise_img,step)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # def _prepare_batch(self, batch):
    #     x, _ = batch
    #     return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        # x = self._prepare_batch(batch)
        x=batch
        noise_img,step,noise_data=self.add_noise(x)
        loss = F.mse_loss(noise_data, self(noise_img,step))
        # self.log(f"{stage}_loss", loss, on_step=True)
        return loss
    
    def sample_image(self,batch_size):
        init_data=torch.randn([batch_size,self.latent_dim]).cuda()
        for i in reversed(range(self.step_num)):
            if i>0:
                noise_data=torch.randn([batch_size,self.latent_dim]).cuda()
            else:
                noise_data=torch.zeros([batch_size,self.latent_dim]).cuda()
            step_data=torch.tensor([i]).repeat(batch_size).cuda()
            predict_noise=self(init_data,step_data)

            alpha_t=self.alpha[i]
            hat_alpha_t=self.hat_alpha[i]
            sigma_t=self.beta[i].sqrt()

            denoise_data=init_data-predict_noise*(1-alpha_t)/torch.sqrt(1-hat_alpha_t)
            denoise_data=denoise_data/alpha_t.sqrt()+sigma_t*noise_data
            init_data=denoise_data
        return init_data



class MyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        DATASETS_PATH=get_dir_path()
        dataset = MNIST(DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,shuffle=True)


def cli_main():
    cli = LightningCLI(
        LitAutoEncoder,
        MyDataModule,
        seed_everything_default=1234,
        run=False,  # used to de-activate automatic fitting.
        trainer_defaults={"callbacks": ImageSampler(), "max_epochs": 10},
        save_config_kwargs={"overwrite": True},
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    print(predictions[0])

def write_image(input_img,img_path):
    a=torchvision.utils.make_grid(
            tensor=input_img
        )
    save_image(a, img_path)




def normalize(input_data):
        data_mean=0.12
        data_std=0.31
        return (input_data-data_mean)/data_std

def get_dataloader(batch_size):
    batch_size=128
    all_data=MyDataModule(batch_size)
    train_data=all_data.train_dataloader()
    eval_data=all_data.test_dataloader()
    return train_data,eval_data


def train_vae():


    vae_net=VAE(28,28,1,128,8,32).cuda()


    
    init_path=r'24_0.5511101597472082_net.pt'
    if init_path is not None and os.path.exists(init_path):
        vae_net.load_state_dict(torch.load(init_path,weights_only=True))

    opt=optim.Adam(vae_net.parameters(),lr=1e-4,weight_decay=1e-4)
    log_dir='logs'
    str_time=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    writer=SummaryWriter(os.path.join(log_dir,str_time))


    run_count=10000
    train_count=0
    eval_count=0

    best_test_loss=None

    def vae_loss(recon_x, x, mu, logvar,kl_weight=1e-2):
        # 计算重构损失，这里使用均方误差（MSE）
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # 计算 KL 散度
        kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence = kl_divergence.mean()
        
        # 总损失为重构损失和 KL 散度的加权和
        total_loss = reconstruction_loss + kl_divergence*kl_weight

        # total_loss=reconstruction_loss
        
        return total_loss
    
    train_data,eval_data=get_dataloader(512)

    for traind_id in range(run_count):
        vae_net.train()
        train_loss=[]
        for id,i in enumerate(train_data):
            data,label=i
            data=data.cuda()
            label=label.cuda()
            data=normalize(data)

            opt.zero_grad()
            recon_data,mean,logvar=vae_net(data)
            loss=vae_loss(recon_data,data,mean,logvar)
            loss.backward()
            opt.step()

            # print(f'training , {traind_id} {id} : {loss.item()}')
            writer.add_scalar('train/loss',loss.item(),train_count)
            train_count+=1

            train_loss.append(loss.item())

            # print(f'max:{data.max()} min:{data.min()} mean:{data.mean()} std:{data.std()}')
            # print(data.shape,label.shape)
            # write_image(data,f'{id}.jpg')

        vae_net.eval()
        eval_loss=[]
        last_eval_input=None
        last_eval_recon=None
        with torch.no_grad():
            for id,i in enumerate(eval_data):
                data,label=i
                data=data.cuda()
                label=label.cuda()
                data=normalize(data)

                recon_data,mean,logvar=vae_net(data)
                loss=vae_loss(recon_data,data,mean,logvar)
                # print(f'test , {traind_id} {id} : {loss.item()}')
                writer.add_scalar('test/loss',loss.item(),eval_count)
                eval_count+=1
                eval_loss.append(loss.item())

                last_eval_input=data
                last_eval_recon=recon_data
        
        train_loss=np.mean(train_loss).item()
        eval_loss=np.mean(eval_loss).item()

        writer.add_scalar('train/loss_all',train_loss,traind_id)
        writer.add_scalar('test/loss_all',eval_loss,traind_id)

        print(f'{traind_id}, train loss:{train_loss}, eval loss:{eval_loss}')
        if best_test_loss is None or eval_loss<best_test_loss:
            torch.save(vae_net.state_dict(), f'{traind_id}_{eval_loss}_net.pt')
            
            save_image(last_eval_input,f'{traind_id}_{eval_loss}_input.jpg')
            save_image(last_eval_recon,f'{traind_id}_{eval_loss}_recon.jpg')
            best_test_loss=eval_loss

class image_encoder:
    def __init__(self,vae_path):
        assert vae_path is not None
        assert os.path.exists(vae_path)

        vae_net=VAE(28,28,1,128,8,32).cuda()
        vae_net.load_state_dict(torch.load(vae_path,weights_only=True))
        vae_net.eval()

        self.vae=vae_net

    @torch.no_grad()
    def encoder(self,input_img):
        mean,_=self.vae.encoder(input_img)
        return mean

    
    @torch.no_grad()
    def decoder(self,input_img):
        out=self.vae.decoder(input_img)
        out=out.reshape(-1,1,28,28)
        return out

    

def train_diffusion():

    latent_dim=32

    duffusion_net=Diffusion(latent_dim,256,8,8,1000).cuda()
    model_light=DiffusionLightning(duffusion_net,latent_dim).cuda()

    vae_class=image_encoder(r'467_0.29906790203686_net.pt')


    def sample_img(img_path):
            model_light.eval()
            model_light.net.eval()
            with torch.no_grad():
                sample_img=model_light.sample_image(64)
                sample_img=vae_class.decoder(sample_img)
                write_image(sample_img,img_path)
    init_path=None
    if init_path is not None and os.path.exists(init_path):
        duffusion_net.load_state_dict(torch.load(init_path,weights_only=True))
    sample_img('diffusion_init_sample.jpg')



    opt=optim.Adam(model_light.net.parameters(),lr=1e-4,weight_decay=1e-4)
    log_dir='logs'
    str_time=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    writer=SummaryWriter(os.path.join(log_dir,f'diffusion_{str_time}'))


    run_count=100
    train_count=0
    eval_count=0

    best_test_loss=None


    def handle_img(input_img):
        input_img=normalize(input_img)
        latent_data=vae_class.encoder(input_img)
        return latent_data
    
    
    train_data,eval_data=get_dataloader(512)

    for traind_id in range(run_count):
        model_light.train()
        model_light.net.train()
        train_loss=[]
        for id,i in enumerate(train_data):
            data,label=i
            data=data.cuda()
            label=label.cuda()
            data=handle_img(data)

            opt.zero_grad()
            loss=model_light.training_step(data,label)
            loss.backward()
            opt.step()

            # print(f'training , {traind_id} {id} : {loss.item()}')
            writer.add_scalar('train/loss',loss.item(),train_count)
            train_count+=1

            train_loss.append(loss.item())

            # print(f'max:{data.max()} min:{data.min()} mean:{data.mean()} std:{data.std()}')
            # print(data.shape,label.shape)
            # write_image(data,f'{id}.jpg')

        model_light.eval()
        model_light.net.eval()
        eval_loss=[]
        with torch.no_grad():
            for id,i in enumerate(eval_data):
                data,label=i
                data=data.cuda()
                label=label.cuda()
                data=handle_img(data)

                loss=model_light.test_step(data,label)
                # print(f'test , {traind_id} {id} : {loss.item()}')
                writer.add_scalar('test/loss',loss.item(),eval_count)
                eval_count+=1
                eval_loss.append(loss.item())
        
        train_loss=np.mean(train_loss).item()
        eval_loss=np.mean(eval_loss).item()

        writer.add_scalar('train/loss_all',train_loss,traind_id)
        writer.add_scalar('test/loss_all',eval_loss,traind_id)

        print(f'{traind_id}, train loss:{train_loss}, eval loss:{eval_loss}')
        if best_test_loss is None or eval_loss<best_test_loss:
            torch.save(model_light.state_dict(), f'diffusion_{traind_id}_{eval_loss}_lightning.pt')
            torch.save(model_light.net.state_dict(), f'diffusion_{traind_id}_{eval_loss}_net.pt')
            
            sample_img(f'diffusion_{traind_id}_{eval_loss}_sample.jpg')
            best_test_loss=eval_loss


if __name__ == "__main__":
    # train_vae()

    train_diffusion()