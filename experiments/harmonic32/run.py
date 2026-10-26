# Adapted for portable reproduction packaging.
# Based on li012589/neuralCT (Apache-2.0), commit f5673f185a2b787f748cb72721eadebf907413a4.
# Changes: portable upstream/output paths and output protection; see docs/PACKAGING.md.
"""32-site fixed-end chain; official CLI settings, corrected Fig.3 evaluation."""
import os, sys, json, time, csv, resource, argparse, copy, platform
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('NEURALCT_RUN_DIR',str(REPO/'runs'/'harmonic32'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
UPSTREAM=Path(os.environ.get('NEURALCT_UPSTREAM',str(REPO/'vendor'/'neuralCT'))).resolve()
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mpl-cache'))
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(UPSTREAM))
import flow, source, utils
from train.learn import reversedKLD
torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float32)
torch.manual_seed(42); np.random.seed(42)
N=32
CFG=dict(n=32,kappa=1,beta=1,num_flow=1,coupling_layers=8,hidden=128,hidden_layers=2,
         batch=200,updates=5000,lr=.001,weight_decay=.001,optimizer='Adam',scheduler=None,
         precision='float32',threads=1,model_seed=42,training_seed=43,evaluation_seed=2026,
         upstream_commit='f5673f185a2b787f748cb72721eadebf907413a4')
def inner(num):
    masks=[]
    for k in range(8):
        if k%2==0:
            b=torch.zeros(num); b[torch.randperm(num)[:num//2]]=1
        else: b=1-b
        masks.append(b.reshape(1,num))
    return flow.RNVP(torch.cat(masks).float(),
        [utils.SimpleMLPreshape([num,128,128,num],[nn.Softplus(),nn.Softplus(),None]) for _ in masks],
        [utils.SimpleMLPreshape([num,128,128,num],[nn.Softplus(),nn.Softplus(),utils.ScalableTanh(num)]) for _ in masks])
def build():
    return utils.flowBuilder(N,1,inner,1,relax=False,shift=False).float()
def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
def save_checkpoint(model,opt,step,elapsed):
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                    train_seconds=elapsed,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),ROOT/'checkpoint.pt')
def train(pilot):
    model=build(); target=source.HarmonicChain(N,1).float()
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.001)
    checkpoint=ROOT/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,weights_only=False,map_location='cpu')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
        start_step=state['step']; elapsed=state['train_seconds']
    else:
        torch.save(model.state_dict(),ROOT/'initial.pt')
        torch.manual_seed(43); start_step=0; elapsed=0.
    end=100 if pilot else 5000
    if start_step>=end: raise RuntimeError('Requested phase already completed; refusing overwrite/retrain')
    start=time.perf_counter()
    with (ROOT/'loss.csv').open('a',newline='') as handle:
        writer=csv.writer(handle)
        if start_step==0: writer.writerow(['update','variational_free_energy','batch_std','elapsed_training_seconds'])
        for k in range(start_step,end):
            loss,sd=reversedKLD(model,target,200)
            opt.zero_grad(); loss.backward(); opt.step()
            total=elapsed+time.perf_counter()-start
            writer.writerow([k+1,loss.item(),sd.item(),total])
            if (k+1)%100==0:
                handle.flush()
                print(json.dumps(dict(update=k+1,loss=loss.item(),train_seconds=total,peak_RSS_MiB=rss_mb())),flush=True)
            if (k+1)%500==0:
                save_checkpoint(model,opt,k+1,total)
                torch.save(model.state_dict(),ROOT/f'model_update_{k+1:04d}.pt')
    total=elapsed+time.perf_counter()-start
    save_checkpoint(model,opt,end,total)
    if pilot:
        # First 100 real training updates, retained for the full run.
        info=dict(updates=100,train_seconds=total,estimated_5000_train_seconds=total*50,
                  peak_RSS_MiB=rss_mb(),threads=1,cpu='Apple M1',memory_GB=8)
        (ROOT/'pilot.json').write_text(json.dumps(info,indent=2))
        print(json.dumps(info,indent=2),flush=True)
    else:
        torch.save(model.state_dict(),ROOT/'trained.pt')
        (ROOT/'training_summary.json').write_text(json.dumps(dict(completed_updates=end,train_seconds=total,peak_RSS_MiB=rss_mb()),indent=2))
def geometry(model,z):
    back=model.forward(model.inverse(z)[0])[0]
    inverse=(back-z).abs().max().item()
    j=torch.zeros(64,64,dtype=z.dtype); j[:N,N:]=torch.eye(N,dtype=z.dtype); j[N:,:N]=-torch.eye(N,dtype=z.dtype)
    result={'inverse_max_abs_error':inverse}
    for name,fn in [('canonical_point',model.layerList[1].inverse),('full_sampler',model.inverse)]:
        errs=[]
        for row in z[:3]:
            m=torch.autograd.functional.jacobian(lambda v: fn(v[None])[0][0],row)
            errs.append((m.T@j@m-j).abs().max().item())
        result[name+'_max_abs_symplectic_defect']=max(errs)
    return result
def evaluate():
    begin=time.perf_counter()
    model=build()
    target=source.HarmonicChain(N,1).double()
    A=target.adj.detach().numpy(); true_cov=np.linalg.inv(2*A)
    logZ=N*np.log(2*np.pi)-.5*np.linalg.slogdet(2*A)[1]
    torch.manual_seed(2026)
    latent=torch.randn(8192,64)
    results={}; samples={}
    for label,file in [('initial','initial.pt'),('final','trained.pt')]:
        model.load_state_dict(torch.load(ROOT/file,weights_only=True,map_location='cpu'))
        vals=[]; arr=[]
        for chunk in latent.split(256):
            x,ld=model.inverse(chunk)
            # Double arithmetic for energy/reduction, transformation remains fp32.
            v=model.prior.logProbability(chunk).double()-ld.double()+target.energy(x.double())+logZ
            vals.append(v.detach().numpy()); arr.append(x.detach().numpy())
        v=np.concatenate(vals); x=np.concatenate(arr); cov=np.cov(x.T)
        results[label]={'reverse_KL_MC':float(v.mean()),'MC_standard_error':float(v.std(ddof=1)/np.sqrt(len(v))),
                        'covariance_relative_Frobenius_error':float(np.linalg.norm(cov-true_cov)/np.linalg.norm(true_cov)),'samples':len(x)}
        samples[label]=x
    diag=model.layerList[0]
    effective=(diag.elements*diag.fix).detach().double().numpy()
    # Formula (4) with unit latent momentum variance; ignore inactive elements.
    omega=np.exp(-effective[:N]); order=np.argsort(omega); omega_sorted=omega[order]
    k=np.arange(1,N+1); analytic=2*np.sin(k*np.pi/(2*N+2))
    results['frequency']={'learned_sorted':omega_sorted.tolist(),'analytic':analytic.tolist(),
         'latent_indices_sorted':order.tolist(),'relative_L2_error':float(np.linalg.norm(omega_sorted-analytic)/np.linalg.norm(analytic)),
         'first_two_relative_errors':((omega_sorted[:2]-analytic[:2])/analytic[:2]).tolist(),
         'effective_log_scales':effective.tolist()}
    x=torch.tensor(samples['final'][:10],requires_grad=True)
    canonical=model.layerList[1]
    Q=canonical.forward(x)[0][:,:N]
    rows=[]; mode_metrics=[]
    for rank,idx in enumerate(order[:2]):
        grad=torch.autograd.grad(Q[:,int(idx)].sum(),x,retain_graph=True)[0][:,:N].detach().double().numpy()
        expected=np.sqrt(2/(N+1))*np.sin(k*(rank+1)*np.pi/(N+1))
        mean=grad.mean(0); sign=1 if np.dot(mean,expected)>=0 else -1
        grad*=sign; mean=grad.mean(0)
        rows.append(grad)
        mode_metrics.append(dict(rank=rank+1,latent_index=int(idx),sign_alignment=sign,
            mean_Jacobian_cosine=float(np.dot(mean,expected)/(np.linalg.norm(mean)*np.linalg.norm(expected))),
            mean_Jacobian_relative_L2_error=float(np.linalg.norm(mean-expected)/np.linalg.norm(expected)),
            across_samples_RMS_std=float(np.sqrt(np.mean(grad.std(axis=0,ddof=1)**2)))))
    results['slow_modes']=mode_metrics
    probe=latent[100:108].clone().requires_grad_()
    results['geometry_float32']=geometry(model,probe)
    shadow=copy.deepcopy(model).double()
    results['geometry_float64_shadow_same_weights']=geometry(shadow,probe.detach().double().requires_grad_())
    results.update(config=CFG,logZ=logZ,evaluation_seconds=time.perf_counter()-begin,
        evaluation_peak_RSS_MiB=rss_mb(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__)
    (ROOT/'metrics.json').write_text(json.dumps(results,indent=2))
    np.savez(ROOT/'evaluation_arrays.npz',latent=latent.numpy(),initial=samples['initial'],final=samples['final'],
             target_covariance=true_cov,learned_frequencies=omega_sorted,analytic_frequencies=analytic,
             sorted_latent_indices=order,slow_mode_Jacobians=np.stack(rows))
    with (ROOT/'frequencies.csv').open('w',newline='') as h:
        w=csv.writer(h); w.writerow(['rank','latent_index_zero_based','learned_frequency','analytic_frequency'])
        w.writerows(zip(k,order,omega_sorted,analytic))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(k,analytic,label='analytic'); axes[0].plot(k,omega_sorted,'o',mfc='none',label='learned'); axes[0].set(xlabel='mode rank k',ylabel='frequency',title='32-site fixed-end chain'); axes[0].legend()
    for r,ax in enumerate(axes[1:]):
        expected=np.sqrt(2/(N+1))*np.sin(k*(r+1)*np.pi/(N+1))
        for line in rows[r]: ax.plot(k,line,'o',mfc='none',alpha=.25,ms=3)
        ax.plot(k,expected,'k-',label='analytic'); ax.plot(k,rows[r].mean(0),label='mean learned')
        ax.set(xlabel='site i',ylabel='dQ / dq_i',title=f'Slow mode rank {r+1} (sign aligned)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT/'fig3_comparison.png',dpi=160); plt.close(fig)
    loss=np.genfromtxt(ROOT/'loss.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(loss['update'],loss['variational_free_energy'],alpha=.22,label='batch loss')
    ax.plot(loss['update'][99:],np.convolve(loss['variational_free_energy'],np.ones(100)/100,'valid'),label='100-update mean')
    ax.axhline(-logZ,color='k',ls='--',label='exact free energy'); ax.set(xlabel='Adam update',ylabel='E[log rho + H]',title='Training (float32, 1 CPU thread)'); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'loss.png',dpi=160); plt.close(fig)
    print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['pilot','train','evaluate']); a=p.parse_args()
    (ROOT/'config.json').write_text(json.dumps(CFG,indent=2))
    if a.phase=='evaluate': evaluate()
    else: train(a.phase=='pilot')
