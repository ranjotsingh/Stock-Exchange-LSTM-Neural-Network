import numpy as np

origfilename = 'training_data.txt'
normfilename = 'training_data_norm.txt'

with open(origfilename, 'r') as f:
    formatdata = f.readline()

x = np.loadtxt(origfilename, usecols=range(0,3), skiprows=1)

in_min = min(x[:,0])
in_max = max(x[:,0])
out_min = min(x[:,2])
out_max = max(x[:,2])

norm = True
if norm:
	if (in_max-in_min) == 0:
		norm = False

if norm:
    normx=x[:,:2]
    normx=(normx-in_min)/(in_max-in_min)
    normy=(x[:,2])[:,None]
    normy=(normy-out_min)/(out_max-out_min)
    normdata=np.concatenate((normx,normy),axis=1)

np.savetxt(normfilename, normdata, '%5.2f')

# Add input/output data to top of norm file
with open(normfilename, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(formatdata.rstrip('\r\n') + '\n' + content)
