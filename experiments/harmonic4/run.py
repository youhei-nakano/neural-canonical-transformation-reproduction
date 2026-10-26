# Adapted for portable reproduction packaging.
# Based on li012589/neuralCT (Apache-2.0), commit f5673f185a2b787f748cb72721eadebf907413a4.
# Changes: portable upstream/output paths and output protection; see docs/PACKAGING.md.
import os, sys, time, json, copy, platform
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
RUN=Path(os.environ.get('NEURALCT_RUN_DIR',str(REPO/'runs'/'harmonic4'))).resolve()
RUN.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(RUN/'mpl-cache'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
UPSTREAM=Path(os.environ.get('NEURALCT_UPSTREAM',str(REPO/'vendor'/'neuralCT'))).resolve()
sys.path.insert(0,str(UPSTREAM))
import flow, source, utils
from train.learn import reversedKLD
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(42); np.random.seed(42)
torch.set_default_dtype(torch.float64)
n=4
def builder(num):
    masks=[]
    for k in range(4):
        if k%2==0:
            b=torch.zeros(num); b[torch.randperm(num)[:num//2]]=1
        else: b=1-b
        masks.append(b.clone())
    return flow.RNVP(torch.stack(masks),
        [utils.SimpleMLPreshape([num,16,16,num],[torch.nn.Softplus(),torch.nn.Softplus(),None]) for _ in masks],
        [utils.SimpleMLPreshape([num,16,16,num],[torch.nn.Softplus(),torch.nn.Softplus(),utils.ScalableTanh(num)]) for _ in masks])
target=source.HarmonicChain(n,1).double()
model=utils.flowBuilder(n,1,builder,1).double()
initial=copy.deepcopy(model.state_dict())
out=RUN/'results'
if out.exists() and any(out.iterdir()):
    raise RuntimeError('Output is not empty; choose a fresh NEURALCT_RUN_DIR')
out.mkdir(exist_ok=True)
torch.manual_seed(2026)
z=model.prior.sample(8192).requires_grad_()
A=target.adj.detach().numpy(); cov=np.linalg.inv(2*A)
logZ=n*np.log(2*np.pi)-.5*np.linalg.slogdet(2*A)[1]
def evaluate():
    x,ld=model.inverse(z)
    v=model.prior.logProbability(z)-ld+target.energy(x)+logZ
    arr=x.detach().numpy()
    return {'reverse_KL_MC':v.mean().item(),'MC_standard_error':v.std().item()/np.sqrt(len(v)),
        'cov_relative_error':float(np.linalg.norm(np.cov(arr.T)-cov)/np.linalg.norm(cov))},arr
before,xb=evaluate()
torch.manual_seed(43)
opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.001)
loss=[]; start=time.perf_counter()
for step in range(600):
    l,_=reversedKLD(model,target,128)
    opt.zero_grad(); l.backward(); opt.step(); loss.append(l.item())
    if step%100==0: print(step,l.item(),flush=True)
elapsed=time.perf_counter()-start
after,xa=evaluate()
torch.save(initial,out/'initial.pt'); torch.save(model.state_dict(),out/'trained.pt')
probe=torch.randn(16,2*n,requires_grad=True)
back=model.forward(model.inverse(probe)[0])[0]
inverse_error=(back-probe).abs().max().item()
J=torch.cat([torch.cat([torch.zeros(n,n),torch.eye(n)],1),torch.cat([-torch.eye(n),torch.zeros(n,n)],1)],0)
point=model.layerList[1]
def defect(fn):
    errs=[]
    for u in probe[:8]:
        M=torch.autograd.functional.jacobian(lambda t: fn(t[None])[0][0],u)
        errs.append((M.T@J@M-J).abs().max().item())
    return max(errs)
metrics={'seed_model':42,'seed_training':43,'seed_evaluation':2026,'steps':600,'batch':128,'n':n,
 'train_seconds':elapsed,'before':before,'after':after,'inverse_max_abs_error':inverse_error,
 'point_symplectic_max_abs_defect':defect(point.inverse),'full_sampler_symplectic_defect':defect(model.inverse),
 'logZ':logZ,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
 'matplotlib':matplotlib.__version__,'threads':torch.get_num_threads()}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
np.savetxt(out/'loss.csv',np.column_stack([np.arange(600),loss]),delimiter=',',header='step,variational_free_energy',comments='')
np.savez(out/'samples.npz',initial=xb,trained=xa,target_covariance=cov)
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
ax[0].plot(loss,alpha=.35,label='batch loss'); ax[0].plot(np.arange(19,600),np.convolve(loss,np.ones(20)/20,'valid'),label='20-step mean'); ax[0].axhline(-logZ,color='k',ls='--',label='exact free energy'); ax[0].legend(fontsize=7); ax[0].set(xlabel='Adam step',ylabel='E[log rho + H]',title='Training')
for a,data,title in zip(ax[1:],[xb,xa],['Initial samples','Trained samples']):
    a.scatter(data[:1500,0],data[:1500,1],s=2,alpha=.2)
    vals,vec=np.linalg.eigh(cov[:2,:2]); theta=np.linspace(0,2*np.pi,200)
    ellipse=vec@np.diag(np.sqrt(vals)*2)@np.array([np.cos(theta),np.sin(theta)])
    a.plot(*ellipse,c='red',label='exact 2-sigma ellipse'); a.set(xlabel='q1',ylabel='q2',title=title,xlim=(-4,4),ylim=(-4,4)); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
print(json.dumps(metrics,indent=2),flush=True)
