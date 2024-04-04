import torch,torchvision
from model import *
if torch.cuda.is_available():
    device = torch.device('cuda:1') #0 is used by others 
else:
    device = torch.device('cpu')
mod = torch.load("sweep.pt", map_location=device) 
#

t = 1
import deepsmiles
from PIL import Image
import json

def decode(x):
  try:
    return converter.decode(x)
  except:
    print("OOBA")
    return x
converter = deepsmiles.Converter(rings=True, branches=True)
def triangle_mask(size):
    mask = 1- np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    mask = torch.autograd.Variable(torch.from_numpy(mask))
    return mask

def top_k_2d(m, k):
  values, indices = torch.topk(m.flatten(), k)
  return indices//m.shape[1], indices%m.shape[1]

def pad_pack(sequences):
    maxlen = max(map(len, sequences))
    batch = torch.LongTensor(len(sequences),maxlen).fill_(0)
    for i,x in enumerate(sequences):
        batch[i,:len(x)] = torch.LongTensor(x)
    return batch, maxlen

reversed_word_map = {}

with open("reverse.map","r") as f:
  reversed_word_map = json.loads(f.read())
reversed_word_map_={}

for i in reversed_word_map.keys():
  reversed_word_map_[int(i)] = reversed_word_map[i]


import numpy as np
def generate(img):
  global mod
  mod = mod.train(False)
  with torch.no_grad():
    mem = mod.decoder.encoder(mod.decoder.encoder_dim(mod.encoder(img)))
    seq = torch.tensor([[77]]).to(device)
    for i in range(100):
      out = mod.decoder.decoder(seq, mem, x_mask=triangle_mask(len(seq)).to(device))[:,-1,:].squeeze(dim=1)
      id = torch.argmax(out, dim=-1)
      if id.item()==78:
        return seq
      seq = torch.concat((seq, id.unsqueeze(0)),dim=-1)

# img1 = torch.tensor(np.array(Image.open("images/"+name).convert("RGB").resize((400,400)))).unsqueeze(0).permute(0,3,1,2).to(device).to(torch.float32)
# ans = generate(img1)
# decode("".join([reversed_word_map_[i] for i in ans[0].numpy()])).replace("<start>","")

import numpy as np
import os
import random
class SMILESGenerator(torch.nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.out = []

  def predict_next(self, i, out, beam, alpha):
    factor = torch.ones(self.top_n_p.shape)
    if i>=1:#forget >=, write factor found this, what causing the reuslt?
      for j in range(beam):
        if self.top_n[j][-1]==78:
          out[j]=torch.zeros(79)
          out[j][78]=1 #0 not 1

        L = len(self.top_n[0])
        try:
          L = (self.top_n[j]==78).nonzero()[0]+1
          # if L.item()==2:
          #   L[0]=torch.inf #yao [0[]]
        except:
          pass
        factor[j] = L

        # print(L)
      # print(factor)
    # print((self.top_n_p.repeat((79,1)).T+torch.log(out)).shape)
    tmp_n = (self.top_n_p.repeat((79,1)).T+torch.log(out))*(1/(factor**alpha)).repeat(79,1).t().to(device)
    # print((1/(factor**alpha)).t().shape)
    # print((self.top_n_p.repeat((79,1)).T+torch.log(out)).shape)
    x, y = top_k_2d(tmp_n, beam)
    # print(x,y)
    new_top_n = []
    new_top_n_p = []
    if i==0:
      factor = torch.ones(beam)*(1/(1**alpha))
      # factor = torch.zeros(beam)*(1/(1**alpha))
    for j in range(beam):
      new_top_n.append(torch.cat((self.top_n[x[j]], y[j].cpu().unsqueeze(-1))))
      new_top_n_p += [tmp_n[x[j], y[j]]/((1/(factor[j]**alpha)))]#return to befpre L
    self.top_n = new_top_n
    self.top_n_p = torch.tensor(new_top_n_p).to(device)


  def forward(self, images, text_in_=[[77]], max_len=100, beam=1,alpha=0.75, method = "sum"): #just changed beam=1 and it runs so beam cannot be random number???  yeah it has to been factor of 676, which is 2 2 13 13 but original paper did not use beam search
    with torch.no_grad():
      image_feature = []
      for i in images:
        mem = self.decoder.encoder(self.decoder.encoder_dim(self.encoder(i)))
        mem = mem.repeat_interleave(beam, dim=0)
        image_feature.append(mem)


      self.top_n = torch.tensor(text_in_)
      self.top_n_p = torch.tensor([0.]).to(device)
      for i in range(max_len):
        padded_text , l = pad_pack(self.top_n)
        padded_text = padded_text.to(device)
        out = []
        for j in range(len(images)):
          out.append(self.decoder.decoder(padded_text, image_feature[j], x_mask=triangle_mask(l).to(device))[:,-1,:].squeeze(dim=1))
        self.out.append(torch.stack(out))
        if method == "sum":
          for j in range(len(out)):
            out[j] = torch.nn.functional.softmax(out[j], dim=1)
          out = torch.sum(torch.stack(out), dim=0)
        else:
          print("BOOOM")
          raise TypeError()
        self.predict_next(i, out, beam, alpha)
        count = 0
        for i in self.top_n:
          if 78 in i:
            count+=1
        if count==beam:
          return self.top_n, self.top_n_p
      return self.top_n, self.top_n_p



mod = mod.eval()
gen = SMILESGenerator(mod.encoder, mod.decoder)
# name = random.choice(val_names)
# print(name)
# img1 = torch.tensor(np.array(Image.open("images/"+name).convert("RGB").resize((400,400)))).unsqueeze(0).permute(0,3,1,2).to(device).to(torch.float32)
# gen = gen.train(False) #forgot gen=
# res = gen.forward([img1], beam=10)
# print()
# for i in res[0]:
  # print(decode("".join([reversed_word_map_[i] for i in i.numpy()])))
# print(torch.exp(res[1]))

import numpy as np
def sample_search(imgs):
  global mod
  gen = mod 
  mod = mod.train(False)
  score = 0
  with torch.no_grad():
    mems = [mod.decoder.encoder(mod.decoder.encoder_dim(mod.encoder(img))) for img in imgs]
    mems = torch.stack(mems, dim=0).squeeze(1)
    # print(mems.shape)
    seq = torch.tensor([[77]]).repeat(len(imgs),1).to(device)
    # print(seq)
    for i in range(100):
      out = mod.decoder.decoder(seq, mems, x_mask=triangle_mask(len(seq[0])).to(device))[:,-1,:]
      sum = np.zeros(79) 
      # print(len(out),out.shape)
      # for i in range(len(out)): 
      for i in range(out.shape[0]): 
        sum+=torch.nn.functional.softmax(out[i], dim=-1).detach().cpu().numpy()
      t = 0.3
      sum = sum/t
      p = (sum/len(out))/np.sum(sum/len(out))
      id = np.random.choice(np.arange(79), p=p) 
      score += np.log(p[id])
      id = torch.tensor([id]).to(device)
      if id.item()==78:
        return seq, score
      seq = torch.concat((seq, id.unsqueeze(0).repeat(len(imgs),1)),dim=-1)
  return seq, score


# print()
# ans = sample_search([img1, img1])[0]
# print(decode("".join([reversed_word_map_[i] for i in ans.cpu().numpy()[0]])))

def myeval(img_paths, beam, method, alpha):
  global gen
  if type(img_paths)!=list:
    img_paths = [img_paths]
  imgs = []
  for img_path in img_paths:
    imgs.append(torch.tensor(np.array(Image.open(img_path).convert("RGB").resize((400,400)))).unsqueeze(0).permute(0,3,1,2).to(device).to(torch.float32))
  gen = SMILESGenerator(mod.encoder, mod.decoder)
  gen = gen.train(False)
  res = gen.forward(imgs, [[77]], 100, beam, alpha=alpha)
  # tmp = [sample_search(imgs) for i in range(beam)]
  # res = [i[0] for i in tmp]
  # scores = [i[1] for i in tmp]
  # top = np.argmax(scores)
  # res[0], res[top] = res[top], res[0]
  # print(res[0][0])
  ans = ""
  for i in range(beam):
    ans+=decode("".join([reversed_word_map_[i.item()] for i in res[0][i]]))+"\n"
    # ans+=decode("".join([reversed_word_map_[i.item()] for i in res[i][0]]))+"\n"

  return ans.replace("<start>","").replace("<end>","")


import pandas as pd
molecules_rows = pd.read_csv("summary.csv")

val_cids = test_cids = [264, 8028, 10952, 68409, 19463, 439846, 11733, 11182, 80290, 18632, 98346, 61929, 12706, 14040, 90803, 23272, 18393, 17149, 16226, 47955, 69814]

new_test_cids = []
for val_cid in val_cids:
  if (len([i for i in os.listdir("./images") if int(i.split("_")[0])==val_cid]))>=5:
    new_test_cids.append(val_cid)

val_cids = new_test_cids
val_names = [i for i in os.listdir("./images") if int(i.split("_")[0]) in val_cids]

beam_ans = []
nImages=1
alpha = 1
print(len(val_cids),len(val_names))
for beam in range(1,5):
  nImageAns = []
  for nImages in range(1,4):
    print(nImages)
    top1_count = 0
    topn_count = 0 
    #not use images number, use id number (19?)*4``
    for val_cid in val_cids:
      val_cid = [val_cid]
      for sample in range(10):
        val_names = list(["images/"+i for i in np.random.choice([i for i in os.listdir("./images") if int(i.split("_")[0]) in val_cid], nImages, replace=False)])
        correct = molecules_rows[molecules_rows[" cid"]==val_cid[0]]["canonicalsmiles"].item()
        predicted = myeval(val_names, beam, "sum", alpha)
        top1_count = top1_count+1 if correct in predicted.split("\n")[:1] else top1_count
        topn_count = topn_count+1 if correct in predicted.split("\n")[:beam] else topn_count
    nImageAns.append((top1_count,topn_count))
  beam_ans.append(nImageAns)
  
print(beam_ans)
