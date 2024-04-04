BATCH_SIZE = 16



import os
if not "images" in os.listdir("."):
  os.system("unzip images.zip")

import sys
import os
import argparse
from tqdm import tqdm
import deepsmiles
from typing import Any, cast, Callable, List, Tuple, Union
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

import sys
sys.path.append("./SwinOCSR/model/Swin-transformer-focalloss")
sys.path.append("./SwinOCSR/model/")
from pre_transformer import Transformer
class FocalLossModelInference:
    """
    Inference Class
    """
    def __init__(self):
        # Load dictionary that maps tokens to integers
        word_map_path = './SwinOCSR/Data/500wan/500wan_shuffle_DeepSMILES_word_map'
        self.word_map = torch.load(word_map_path)
        self.inv_word_map = {v: k for k, v in self.word_map.items()}

        # Define device, load models and weights
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        # self.args, config = self.get_inference_config()
        # self.encoder = build_model(config, tag=False)
        self.decoder = self.build_decoder()
        # self.load_checkpoint("./swin_transform_focalloss.pth")
        self.decoder = self.decoder.to(self.dev).eval()
        # self.encoder = self.encoder.to(self.dev).eval()

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint and update encoder and decoder accordingly

        Args:
            checkpoint_path (str): path of checkpoint file
        """
        print(f"=====> {checkpoint_path} <=====")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # encoder_msg = self.encoder.load_state_dict(checkpoint['encoder'],
        #                                            strict=False)
        decoder_msg = self.decoder.load_state_dict(checkpoint['decoder'],
                                                   strict=False)
        # print(f"Encoder: {encoder_msg}")
        print(f"Decoder: {decoder_msg}")
        del checkpoint
        torch.cuda.empty_cache()

    def build_decoder(self):
        """
        This method builds the Transformer decoder and returns it
        """
        self.decoder_dim = 256  # dimension of decoder RNN
        self.ff_dim = 2048
        self.num_head = 8
        self.dropout = 0.1
        self.encoder_num_layer = 6
        self.decoder_num_layer = 6
        self.max_len = 277
        self.decoder_lr = 5e-4
        self.best_acc = 0.
        return Transformer(dim=self.decoder_dim,
                           ff_dim=self.ff_dim,
                           num_head=self.num_head,
                           encoder_num_layer=self.encoder_num_layer,
                           decoder_num_layer=self.decoder_num_layer,
                           vocab_size=len(self.word_map),
                           max_len=self.max_len,
                           drop_rate=self.dropout,
                           tag=False)
transformer_ = FocalLossModelInference()

import base64
import pandas as pd
import os

import sys
sys.path.append("./SwinOCSR/model/Swin-transformer-focalloss")
sys.path.append("./SwinOCSR/model/")

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')


eff = torchvision.models.efficientnet_v2_s()
mynet = eff.features
class ImageEncoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.eff = mynet.to(device)
    self.projection = torch.nn.Linear(1280,256).to(device)
  def forward(self, images):
    features = self.eff(images)
    # print(features.shape)
    features = torch.flatten(features, start_dim=2, end_dim=3)
    features = torch.permute(features, (0, 2, 1))
    return self.projection(features)



class Image2SMILES(torch.nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    # self.decoder.encoder_dim = torch.nn.Dropout(dropout)

  def forward(self, image, text_in, xmask, dropout):
    image_feature = self.encoder(image)
    out = self.decoder(text_in, torch.nn.functional.dropout(image_feature, dropout,training=self.training), x_mask=xmask)
    return out

mod = Image2SMILES(ImageEncoder(), transformer_.decoder)

mod = torch.load("256.pt", map_location=device)

# torch.save( mod.encoder, "drive/MyDrive/encoder")

def pad_pack(sequences):
    maxlen = max(map(len, sequences))
    batch = torch.LongTensor(len(sequences),maxlen).fill_(0)
    for i,x in enumerate(sequences):
        batch[i,:len(x)] = torch.LongTensor(x)
    return batch, maxlen

torch.topk(torch.tensor([1,2,3,4]),2)

def triangle_mask(size):
    mask = 1- np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    mask = torch.autograd.Variable(torch.from_numpy(mask))
    return mask

reversed_word_map = {}
import json
with open("reverse.map","r") as f:
  reversed_word_map = json.loads(f.read())

from focal_loss.focal_loss import FocalLoss
m = torch.nn.Softmax(dim=-1)
lf = FocalLoss(gamma=2, ignore_index=0)#torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")
def loss_fn(pred, truth):
  pred = m(pred)
  l = lf(pred, truth)
  return l

def mask_acc(pred, truth):
    pred = torch.argmax(pred, -1)
    mask = truth != 0
    match_case = truth == pred
    return torch.sum(mask*match_case)/torch.sum(mask)

import pandas as pd
molecules_rows = pd.read_csv("summary.csv")
val_smiles = np.random.choice(list(set(molecules_rows["canonicalsmiles"])), size=20, replace=False)
val_cids = list(molecules_rows[molecules_rows["canonicalsmiles"].isin(val_smiles)][" cid"])
print(val_cids)
import random
converter = deepsmiles.Converter(rings=True, branches=True)
cids = list(molecules_rows[" cid"])
train_cids = [i for i in cids if not i in val_cids]
val_cids = val_cids
test_cids = []


train_names = [i for i in os.listdir("./images") if int(i.split("_")[0]) in train_cids]
val_names = [i for i in os.listdir("./images") if int(i.split("_")[0]) in val_cids]
from PIL import ImageOps

def getitem(index, train=True):
  global val_names, train_names
  ti = []
  to = []
  imgs = []

  # train=FalseTAMADE WEISM TRAIN = FALSE!!!guaibude bunengfanhua shouzhima ghuainbbude val namehao zuibakouke azhiqianweismkeyile? jiushiyingweimeiyouyongnamesle
  names = train_names if train else val_names
  start = index*BATCH_SIZE
  end =  (index+1)*BATCH_SIZE
  end = len(names) if end>len(names) else end
  # print(start, end)
  for i in range(start, end):


    name = random.choice([i for i in names])
    # name = random.choice([i for i in os.listdir("./images")]) #shetoudzikunhzhegegebuhaishisuijidemaguaibude

    img = Image.open(f"images/{name}").convert("RGB")
    if train:
      img = img.rotate(random.choice([0,90,180,270]), expand=0).resize((400,400))
      if random.random()>0.5:
        img = ImageOps.flip(img)
      if random.random()>0.5:
        img = ImageOps.mirror(img)
      img.crop((random.random()*30,random.random()*30,400-random.random()*30,400-random.random()*30))
    img = np.array(img.resize((400,400)), dtype="float32")
    if train:
      # img[:,:,0] *= random.random()*0.2+0.9
      img[:,:,1] *= random.random()*0.1+0.9
      img[:,:,2] *= random.random()*0.1+0.9
      img[:,:,:] *= random.random()*0.1+0.8
    img = torch.tensor(np.array(img)).permute(2, 0, 1).to(torch.float32).to(device)
    # if train:
    #   # img+=torch.rand((3,400,400)).to(device)*50
    #   for _ in range(30):
    #     x,y = int(random.random()*380),int(random.random()*380)
    #     img[:,x:x+20, y:y+20] = torch.rand((3,20,20))*250
#  guoranhaishiyaoqudizokunduzikoue
    # if train:
      # img*=torch.round(torch.tanh(5*torch.rand((3,400,400)))).to(device)
    # print(name)
    # print(converte//r.encode(molecules_rows[molecules_rows[" cid"]==int(name.split("_")[0])]["canonicalsmiles"].item()))
    smiles = [transformer_.word_map[i] for i in converter.encode(molecules_rows[molecules_rows[" cid"]==int(name.split("_")[0])]["canonicalsmiles"].drop_duplicates(keep="first").item())]
    ti.append([77] + smiles)
    to.append(smiles + [78])
    imgs.append(img)
  return torch.stack(imgs), ti, to


print(f"Number of images {len(val_names)}")
def val():
  global val_names
  # global mod
  np.random.shuffle(val_names)
  running_loss = 0
  eval_mod = mod.train(False).to(device)  #forget to use =
  # ids = os.listdir("images")[:10]

  valacc = []
  for i in range(len(val_names)//(BATCH_SIZE)):
    image, text_in, text_out = getitem(i, train=False)

    image = image.to(device)
    text_out = pad_pack(text_out)[0].to(device)
    padded_x = pad_pack(text_in)

    xmask = triangle_mask(padded_x[1]).to(device)
    text_in = padded_x[0].to(device)

    outputs = eval_mod(image, text_in, xmask, 0)
    loss = loss_fn(outputs, text_out)



    running_loss += loss.item()
    valacc.append(mask_acc(outputs.detach(), text_out).item())
  # return outputs
  return running_loss, np.mean(valacc)
import gc
mod.cpu()
gc.collect()
torch.cuda.empty_cache()
mod = torch.load("256.pt", map_location=device) #it is model not dict
for i in mod.decoder.parameters():
  i.require_grad = False
for i in mod.encoder.eff[6:].parameters():
  i.require_grad = False

# for layers in mod.encoder.eff[:4]:
#   for i in layers.parameters():
#     i.require_grad = False

def train(reload=True, split=1, pretrain="mol_80k"):
  global mod
#   config={
#     "dropout":0.1877397430344046,
#     "epoch":4,
#     "gamma":0.9865237150288334,
#     "lr":0.0005572473602051692
#     }
  wandb.init()
  lr, gamma, epoch, dropout = wandb.config["lr"],  wandb.config["gamma"], wandb.config["epoch"], wandb.config["dropout"]
  mod = torch.load("256.pt", map_location=device)
  if reload:
    mod = torch.load("256.pt", map_location=device) #it is model not dict
    if pretrain == "image_net":
      mod.encoder.projection.reset_parameters()
      mod.encoder.eff = torchvision.models.efficientnet_v2_s(weights="DEFAULT").features.to(device)
    elif pretrain == "none":
      mod.encoder.projection.reset_parameters()
      mod.encoder.eff = torchvision.models.efficientnet_v2_s().features.to(device)
    for i in mod.parameters():
      i.require_grad = True
    for i in mod.decoder.parameters():
      i.require_grad = False

    for i in mod.encoder.parameters():
      i.require_grad = False
    for i in mod.encoder.eff.parameters():
      i.require_grad = True
    for i in mod.encoder.projection.parameters(): #forgot about this forgrt to set oiit to false just checked history this is why
      i.require_grad = True
  print(lr, gamma, epoch, dropout)
  losses = []
  access = []
  val_acc = []
  val_loss = []
  optimizer = torch.optim.AdamW(
    mod.parameters(),
    lr=lr)
  import pylab
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
  running_loss = 0
  for _ in range(epoch): #why there is no this problem before??!?!?!
    np.random.shuffle(train_names)
    mod = mod.train(True) #this will cuz a deep in training loss
    # ids = [i for i in os.listdir("images") if not i.startswith("638066")]
    # ids = os.listdir("images")[10:]
    # np.random.shuffle(ids)
    for i in range(len(train_names)//(BATCH_SIZE*split)):#why set this and it be better or is it learning rate? it lower faster? hyper-parameter kunduzikouke gaile lenoslistdir dao trainnames cjiuhuichaoguo100?wtfmeiyouyunxingduzikunk
      if i==None:
        continue
      mod = mod.train(True)

      start_index = i
      image, text_in, text_out = getitem(i, True)
      image = image.to(device) #mutli process cannot use cuda so moved here
      # image = torch.permute(image, (0, 3, 1, 2))
      text_out = pad_pack(text_out)[0].to(device)
      padded_x = pad_pack(text_in)

      xmask = triangle_mask(padded_x[1]).to(device)
      text_in = padded_x[0].to(device)

      optimizer.zero_grad()
      outputs = mod(image, text_in, xmask, dropout)
      #loss = loss_fn(outputs, text_outi) guaibude yyixiazinamegao
      loss = loss_fn(outputs, text_out)

      loss.backward()

      optimizer.step()

      # losses.append(loss.item())
      wandb.log({"loss":loss.item()})
      wandb.log({"acc":mask_acc(outputs.detach(), text_out)})

      if i%20==0:
        a, b = val()
        val_loss.append(a)
        val_acc.append(b)
        wandb.log({"val_loss":a})
        wandb.log({"val_acc":b})
        scheduler.step()
  # pylab.plot([i.item() for i in access], label="Train")
  # pylab.plot(np.arange(1, len(val_acc)+1)*(len(access)/len(val_acc)),val_acc,label="Val")
  # pylab.legend()
  # pylab.show()
  # # pylab.savefig(f"{lr}_{gamma}_{epoch}_{run_num}.png")
  # return mod, access, losses, val_acc, val_loss
  

reversed_word_map_={}
for i in reversed_word_map.keys():
  reversed_word_map_[int(i)] = reversed_word_map[i]

np_map = np.array(list(reversed_word_map_.values()))
for i in reversed_word_map_.keys():
  np_map[i] = reversed_word_map_[i]

def top_k_2d(m, k):
  values, indices = torch.topk(m.flatten(), k)
  return indices//m.shape[1], indices%m.shape[1]
  # return indices//m.shape[0]-1, indices%m.shape[1]

import wandb
wandb.login(key="eacfc8100234423e39351d1876a55a9c4f6a9290")
# train()
# torch.save(mod, "sweep.pt")
# import wandb
# wandb.login()
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "lr": {"max": 0.001, "min": 0.0005},
        "gamma": {"max": 1.0, "min": 0.95},
        "epoch": {"values":[1,2,3,4]},
        "dropout": {"min":0.0, "max":0.5}
    },
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="length prediction sweep")
wandb.agent(sweep_id, function=train)