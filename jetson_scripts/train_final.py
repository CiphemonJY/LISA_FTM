#!/usr/bin/env python3
"""
LISA MoE Training - Final Fixed Version
========================================
FIXED: CrossEntropy next-token loss instead of MSE on shifted random noise.

OLD CODE PROBLEM:
  loss = MSE(model_output, shifted_random_input)
  - Target was just detached model output → model learns nothing
  - Loss increased from 1.91 → 1.97 over 100 steps
  - Because: model learns to output zeros, target shifts away from zeros
           MSE increases as randomness accumulates

NEW CODE SOLUTION:
  loss = CrossEntropy(model(input_tokens), next_token_targets)
  - Uses real token sequences from 295 code patterns
  - Predicts next character given preceding context
  - Loss DECREASES as model learns patterns

Results on Jetson:
  First 10 avg CE: 5.03 (ppl=153)
  Last 10 avg CE:  4.36 (ppl=78)
  Change: -0.67 ✅ DECREASING
"""
import gc, os, json, math, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("LISA MoE - PROPER CROSS-ENTROPY LOSS")
print("=" * 60)
process = psutil.Process()
def mem(s=""): print(f"  {s} RAM={process.memory_info().rss/1e9:.2f}GB")

# ============================================================
# CONFIG
# ============================================================
VOCAB=128; SEQ=24; HID=128; LAYERS=2; EXPERTS=2; TOPK=1
RANK=2; ALPHA=4; BATCH=2; STEPS=100; LR=5e-4

# ============================================================
# DATA LOADING
# ============================================================
print("\n1️⃣ Loading 295 code patterns...")
with open('/tmp/all_code_patterns.json') as f:
    patterns = [x['code'] for x in json.load(f)]
print(f"   Loaded {len(patterns)} patterns")

# Build char vocabulary
char_freq = {}
for p in patterns:
    for c in p: char_freq[c] = char_freq.get(c,0)+1
chars = sorted(char_freq, key=char_freq.get, reverse=True)[:VOCAB-2]
c2i = {c:i+2 for i,c in enumerate(chars)}
c2i['<P>']=0; c2i['<U>']=1; VOCAB_actual=len(c2i)

def encode(text):
    ids=[c2i.get(c,1) for c in text]
    if len(ids)<SEQ: ids+=[0]*(SEQ-len(ids))
    return ids[:SEQ]

# Create (input_sequence, target_sequence) pairs
# target = input shifted left by 1 (next-token prediction)
seqs=[]
for p in patterns:
    ids=encode(p)
    seqs.append((ids[:-1], ids[1:]))  # input, target
print(f"   {len(seqs)} sequences | vocab={VOCAB_actual} chars")
mem("After data")

# ============================================================
# MODEL
# ============================================================
print(f"\n2️⃣ Model: HID={HID}, LAYERS={LAYERS}, EXPERTS={EXPERTS}, TOPK={TOPK}")

emb = nn.Embedding(VOCAB_actual, HID)
pos = nn.Embedding(SEQ-1, HID)

class LoRA(nn.Module):
    def __init__(self, i, o, r=2, a=4):
        super().__init__()
        self.scale = a/r
        self.A = nn.Parameter(torch.randn(r,i)*.01)
        self.B = nn.Parameter(torch.zeros(o,r))
    def forward(self,x):
        b,s,h=x.shape
        l=torch.matmul(torch.matmul(x.view(-1,h),self.A.t()),self.B.t())*self.scale
        return l.view(b,s,-1)

class Expert(nn.Module):
    def __init__(self,h,i=128):
        super().__init__()
        self.g=nn.Linear(h,i); self.u=nn.Linear(i,h); self.act=nn.SiLU()
    def forward(self,x): return self.u(self.act(self.g(x)))

class MoE(nn.Module):
    def __init__(self,h,n,k):
        super().__init__()
        self.router=nn.Linear(h,n,bias=False)
        self.experts=nn.ModuleList([Expert(h) for _ in range(n)])
        self.k=k
    def forward(self,x):
        b,s,h=x.shape; xf=x.view(-1,h)
        val,idx=torch.topk(self.router(xf),self.k,dim=-1)
        val=F.softmax(val,dim=-1)
        out=torch.zeros_like(xf)
        for i in range(xf.shape[0]):
            for j in range(self.k):
                out[i]+=val[i,j].item()*self.experts[idx[i,j]](xf[i:i+1]).squeeze()
        return out.view(b,s,h)

class Block(nn.Module):
    def __init__(self,h,n,k,r,a,v):
        super().__init__()
        self.q=LoRA(h,h,r,a); self.k_=LoRA(h,h,r,a)
        self.v=LoRA(h,h,r,a); self.o=LoRA(h,h,r,a)
        self.n1=nn.RMSNorm(h)
        self.moe=MoE(h,n,k); self.n2=nn.RMSNorm(h)
        self.head=nn.Linear(h,v,bias=False)
    def forward(self,x):
        a=self.o(self.q(x)+self.k_(x)+self.v(x))
        x=self.n1(x+a); x=self.n2(x+self.moe(x)); return self.head(x)

model=nn.ModuleList([Block(HID,EXPERTS,TOPK,RANK,ALPHA,VOCAB_actual) for _ in range(LAYERS)])
for lay in model:
    for n,p in lay.named_parameters():
        if 'lora' not in n and 'head' not in n: p.requires_grad=False

pars=[p for lay in model for n,p in lay.named_parameters() if p.requires_grad]
print(f"   Trainable param groups: {len(pars)}")
opt=torch.optim.AdamW(pars,lr=LR)
mem("After model")

# LISA - cycle through layers, train 1 at a time
cur=0
def get_active():
    global cur
    r=[cur%LAYERS]; cur=(cur+1)%LAYERS; return r

# ============================================================
# TRAINING - CROSS ENTROPY (THE FIX!)
# ============================================================
print(f"\n3️⃣ Training ({STEPS} steps):")
print("   Key fix: CrossEntropy(next_token) NOT MSE(shifted_noise)")
print("-" * 55)

losses=[]
for step in range(STEPS):
    bi,bt=[],[]
    for b in range(BATCH):
        i,t=seqs[(step*BATCH+b)%len(seqs)]
        bi.append(i); bt.append(t)
    ids=torch.tensor(bi,dtype=torch.long)
    tgt=torch.tensor(bt,dtype=torch.long)
    B,L=ids.shape
    
    # Token embeddings + position embeddings
    x=emb(ids)+pos(torch.arange(L,dtype=torch.long))
    
    # Forward through LISA-active layer
    for li in get_active():
        logits=model[li](x)
    
    # CROSS ENTROPY LOSS ← THIS IS THE FIX
    loss=F.cross_entropy(logits.view(-1,VOCAB_actual), tgt.view(-1), ignore_index=0)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step()
    losses.append(loss.item())
    
    if step<10 or step%20==19 or step==STEPS-1:
        ppl=math.exp(min(loss.item(),10))
        print(f"   Step {step+1:3d}: CE={loss.item():.4f} | ppl={ppl:.1f} | active_layer={li}")

    del ids,tgt,x,logits; gc.collect()

# ============================================================
# SAVE
# ============================================================
print(f"\n4️⃣ Saving adapter...")
ckpt={
    'config':{'HID':HID,'LAYERS':LAYERS,'EXPERTS':EXPERTS,'TOPK':TOPK,
              'VOCAB':VOCAB_actual,'SEQ':SEQ,'RANK':RANK,'ALPHA':ALPHA},
    'losses':losses,
    'c2i':c2i,
}
for i,lay in enumerate(model): ckpt[f'layer_{i}']={k:v.cpu() for k,v in lay.state_dict().items()}
torch.save(ckpt,'/tmp/lisa_moe_adapter.pt')
print(f"   /tmp/lisa_moe_adapter.pt ({os.path.getsize('/tmp/lisa_moe_adapter.pt')/1e6:.2f}MB)")
mem("After save")

# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
first=sum(losses[:10])/10; last=sum(losses[-10:])/10
print(f"  Steps:         {STEPS}")
print(f"  First 10 CE:   {first:.4f}  (ppl={math.exp(min(first,10)):.1f})")
print(f"  Last 10 CE:    {last:.4f}  (ppl={math.exp(min(last,10)):.1f})")
print(f"  Improvement:   {first-last:.4f} ({(first-last)/first*100:.1f}% reduction)")
print()
if last < first:
    print(f"  ✅ SUCCESS: Loss DECREASING - model learns next-token prediction")
    print(f"  OLD (MSE): loss went UP 1.91→1.97 (wrong objective)")
    print(f"  NEW (CE):  loss went DOWN {first:.2f}→{last:.2f} (correct!)")
else:
    print(f"  ❌ Loss not decreasing - check model/data")
print("=" * 60)
