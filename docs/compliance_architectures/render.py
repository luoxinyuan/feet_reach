"""Render editable Graphviz relationship code. Usage: python render.py [--dot /path/to/dot]."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dot',default='dot');a=p.parse_args();out=Path(__file__).resolve().parent
header='''digraph G {
graph [rankdir=LR,bgcolor="#ffffff",pad="0.35",nodesep="0.45",ranksep="0.65",fontname="DejaVu Sans",fontsize=20,labelloc=t];
node [shape=box,style="rounded,filled",fillcolor="#e9f2ff",color="#4977a8",fontname="DejaVu Sans",fontsize=12,margin="0.16,0.12"];
edge [color="#4d6278",fontname="DejaVu Sans",fontsize=10,arrowsize=0.8];
'''
def diagram(name,title,nodes,edges):
 s=header.replace('rankdir=LR','rankdir=TB' if name in ('00_overview','05_force_estimator') else 'rankdir=LR')+'label='+json.dumps(title)+';\n'
 for key,label,kind in nodes:
  color={'train':'#fff0d6','frozen':'#e5e7eb','motion':'#ede8ff','sim':'#dcf5e7'}.get(kind,'#e9f2ff')
  s+=key+' [label='+json.dumps(label)+',fillcolor="'+color+'"];\n'
 for source,target,label,training in edges:
  s+=f'{source} -> {target} [label={json.dumps(label)}'+(',style=dashed,color="#bd791f",constraint=false' if training else ',constraint=false' if (source,target) in [('robot','obs'),('robot','state'),('robot','hist'),('integrate','obs')] else '')+'];\n'
 s+='}\n';path=out/(name+'.dot');path.write_text(s)
 for fmt in ['svg','png','pdf']:subprocess.run([a.dot,'-T'+fmt,str(path),'-o',str(out/(name+'.'+fmt))],check=True)
# Dashed arrows indicate training-only supervision, not differentiating through physics.
diagram('00_overview','Compliance implementations: architecture vs. force generator',[
 ('data','Motion dataset\nAMASS / early Crocoddyl','motion'),('force','Force generator (independent axis)\nTransition / frame spring / online J / offline segmented J','motion'),
 ('sim','Robot simulator\nactual state + physically applied force','sim'),('direct','A/B: Direct action policies\nA: privileged teacher -> adaptation -> student\nB: target-free actor + privileged critic',''),
 ('res','C: Residual with GT force\nFrozen SONIC or CEER',''),('amp','D: Residual + AMP\nC plus training-only discriminator','train'),('est','E: Residual with estimated force\nGT teacher -> distill -> fine-tune',''),('reward','Dense tracking reward + critic\nReference motion is training-only','train')],
 [('data','force','reference trajectory',False),('force','sim','physical force',False),('sim','direct','state + GT force',False),('sim','res','state + GT force',False),('res','amp','same deployment structure',False),('sim','est','state/action history; GT supervision',True),('data','reward','targets',True),('reward','direct','PPO',True),('reward','res','PPO',True),('reward','est','PPO',True)])
diagram('01_direct_teacher_student','A | Direct action: privileged teacher -> adaptation -> student',[
 ('state','Public state/history + GT force',''),('target','Training-only privileged state\nand motion joint target','motion'),('enc','Privileged encoder\n+ optional direct target channel','train'),('teacher','Teacher actor\nPPO dense tracking','train'),('adapt','Adaptation MLP\npredict privileged latent\n(optional target prediction)',''),('student','Student actor\nno external joint target',''),('robot','Joint-position action -> PD -> robot\nNo frozen motion tracker','sim'),('loss','Latent / optional direct-target MSE\nthen student PPO fine-tuning','train')],
 [('target','enc','',True),('enc','teacher','latent / optional direct target',True),('state','teacher','',True),('state','adapt','',False),('adapt','student','predicted latent / optional target',False),('state','student','',False),('teacher','student','copy actor weights',True),('enc','loss','latent supervision',True),('loss','adapt','train',True),('loss','student','train',True),('teacher','robot','phase 1 rollout action',True),('student','robot','joint actions',False)])
diagram('02_direct_ppo','B | Direct target-free actor with dense tracking PPO',[
 ('data','Motion reference','motion'),('gen','Guidance generator','motion'),('obs','Actual state/history + GT force',''),('actor','Trainable direct-action actor',''),('robot','Joint-position action -> PD -> robot','sim'),('reward','Dense tracking reward\nprivileged critic','train')],
 [('data','gen','',False),('gen','robot','external force',False),('robot','obs','',False),('obs','actor','no motion target',False),('actor','robot','actions',False),('data','reward','',True),('robot','reward','actual motion',True),('reward','actor','PPO',True)])
diagram('03_residual_gt','C | GT-force residual actor over a frozen motion tracker',[
 ('data','Motion reference\n(training only)','motion'),('gen','Spring / online Jacobian /\noffline segmented generator','motion'),('robot','Robot + PD actuators','sim'),('obs','Proprioception + actual GT force\n+ own q_ref / tracking error / ref velocity',''),('actor','Trainable residual actor\n29 joint-reference increments',''),('integrate','Reference integrator\nq_ref += dt * speed * tanh(action)\nlimits + tracking-error bound',''),('low','FROZEN CEER or SONIC\nCEER: bypass joint predictor\nSONIC: frozen encoder/decoder path','frozen'),('reward','Dense tracking + regularization\nprivileged critic','train')],
 [('data','gen','',False),('gen','robot','external force',False),('robot','obs','state + applied force',False),('obs','actor','',False),('actor','integrate','residual',False),('integrate','low','self-generated reference',False),('low','robot','low-level action',False),('integrate','obs','own reference (not dataset target)',False),('data','reward','target',True),('robot','reward','actual state',True),('reward','actor','PPO',True)])
diagram('04_residual_amp','D | Dense tracking + AMP motion-prior ablation',[
 ('real','AMASS expert state pairs','motion'),('fake','Robot rollout state pairs','sim'),('disc','Trainable discriminator\nexpert vs rollout/replay\nclassification + gradient penalty','train'),('style','AMP style reward','train'),('dense','Dense tracking reward','train'),('ppo','Residual PPO actor update','train'),('actor','Residual actor\nstate + GT force + own reference',''),('low','Reference integrator\n+ FROZEN CEER','frozen'),('robot','Robot simulator','sim')],
 [('real','disc','real',True),('fake','disc','fake',True),('disc','style','score',True),('style','ppo','bonus (warm-up)',True),('dense','ppo','retained',True),('ppo','actor','update',True),('actor','low','residual',False),('low','robot','action',False),('robot','fake','',True)])
diagram('05_force_estimator','E | Force-estimated residual student: three training phases',[
 ('gt','GT physically applied force\nroot frame / 30 N\ntraining only','motion'),('state','Public proprioception\n+ own reference/error',''),('hist','10 causal frames x 93\nangular velocity / gravity / q / dq\nactual low-level action',''),('est','Force estimator\nMLP 930 -> 512 -> 256 -> 3K\nK=3 or 23',''),('teacher','1. GT-force residual teacher\n4B dense PPO; then frozen','train'),('student','2. Distill (1B): student rollout\n3. Fine-tune (4B): estimated-force PPO\n+ teacher action imitation',''),('loss','Force MSE supervision\nEstimator updated AFTER PPO','train'),('low','Reference integrator\n+ FROZEN general CEER','frozen'),('robot','Robot simulator\nonline J (3kp) / cached delta force (23-link)','sim'),('reward','Motion targets -> dense reward\nand privileged critic ONLY','train')],
 [('state','teacher','',True),('gt','teacher','GT',True),('hist','est','',False),('est','student','estimated force',False),('state','student','',False),('teacher','student','action labels on student states',True),('gt','loss','label',True),('est','loss','prediction',True),('loss','est','update',True),('teacher','low','phase 1 residual rollout',True),('student','low','residual',False),('low','robot','action',False),('robot','hist','causal measurements',False),('robot','state','',False),('reward','teacher','PPO phase 1',True),('reward','student','PPO phase 3',True)])
print(out)
