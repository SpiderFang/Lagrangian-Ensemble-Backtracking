import sys,json
sys.path.insert(0,'/home/mustlab/Workspace/lbt-wp75/fullpage'); sys.path.insert(0,'/home/mustlab/Workspace/lbt-wp75/combined_proto_src3')
import numpy as np
import cp_lib as C
S=C.L0.prep_site('hsinchu'); sd=S.sd
ci=sd.c_idx; m=ci>=0
x=sd.cols['coast_arrival_x_m'][m]; y=sd.cols['coast_arrival_y_m'][m]
lo,la=sd.proj.unproject(np.array(x,float),np.array(y,float))
out=[[int(i),float(a),float(b)] for i,a,b in zip(ci[m],lo,la)]
json.dump(out,open('/home/mustlab/hs_arr.json','w'))
print(len(out))
