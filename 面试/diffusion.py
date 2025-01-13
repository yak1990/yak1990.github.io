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

from lightning.pytorch import LightningDataModule, LightningModule, Trainer, callbacks
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms
    from torchvision.utils import save_image

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

class Diffusion(nn.Module):
    def __init__(self,H,W,C,hidden_dim,group_num,layout_num,step_num,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h=H
        self.w=W
        self.c=C

        self.pre_net=nn.Sequential(
            nn.Conv2d(C,hidden_dim,kernel_size=3,padding=1),
            nn.GroupNorm(group_num,hidden_dim),
            nn.LeakyReLU(),
        )
        mid_net_list=[]
        for i in range(layout_num):
            mid_net_list.append(nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1))
            mid_net_list.append(nn.GroupNorm(group_num,hidden_dim))
            mid_net_list.append(nn.LeakyReLU())
            mid_net_list.append(nn.Dropout2d())
        self.mid_net=nn.Sequential(
            *mid_net_list
        )
        self.post_net=nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1),
            nn.GroupNorm(group_num,hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim,C,kernel_size=3,padding=1)
        )

        self.pos_params=nn.Parameter(torch.rand(step_num,hidden_dim))
        
        

    def forward(self,input_data,step):
        hidden_data=self.pre_net(input_data)

        B,C,H,W=hidden_data.shape
        pos_data=self.pos_params[step].view(B,C,1,1)
        hidden_data2=hidden_data+pos_data

        hidden_data3=self.mid_net(hidden_data2)
        out=self.post_net(hidden_data3)
        return out

class DiffusionLightning(LightningModule):
   

    def __init__(self, H,W,C,hidden_dim,group_num,layout_num,step_num):
        super().__init__()
        self.save_hyperparameters()
        self.step_num=step_num
        self.net=Diffusion( H,W,C,hidden_dim,group_num,layout_num,step_num)
        self.init_diffusion_params()
    
    def init_diffusion_params(self):
        beta=1/self.step_num

        self.beta=torch.tensor([beta for _ in range(self.step_num)])
        self.alpha=1-self.beta

        self.hat_alpha=torch.cumprod(self.alpha)



    def add_noise(self,input_data):
        B=input_data.shape[0]
        noise_data=torch.randn_like(input_data)
        step_data=torch.randint(0,self.step_num,(B,))
        hat_alpha=self.hat_alpha[step_data].view(B,1,1,1)
        img_corr=torch.sqrt(self.hat_alpha)
        noise_corr=1-hat_alpha

        out_img=img_corr*input_data+noise_corr*noise_data
        return out_img



    def forward(self, input_data,input_step):
        return self.net(input_data,input_step)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss


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
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


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

def main():
    # a=Diffusion(64,64,1,128,8,3,100)
    # b=torch.rand((8,1,64,64))
    # c=torch.tensor([1,2,3,4,5,6,7,8])
    # d=a(b,c)
    # print(d.shape)

    a=MyDataModule()
    b=a.train_dataloader()
    for i in b:
        data,label=i
        print(data.shape,label.shape)

if __name__ == "__main__":
    # cli_main()
    main()