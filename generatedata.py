from random import randint

with open('norm.dat', 'r') as f:
    norm_line = f.readline()
    norm_line

norm = True if  == 'y' else False
trainOptions = f.readline().split()
numInputs = (int)(trainOptions[trainOptions.index('Inputs:')+1])
numOutputs = (int)(trainOptions[trainOptions.index('Outputs:')+1])

name="training_data.txt"
file=open(name,'w')
file.write('Inputs: 2 Outputs: 1\n')

min = 0.0
max = 10.0
tmax = max+max
if norm:
	if (max-min) == 0:
		norm = False
	elif (tmax-min) == 0:
		norm = False
	
for i in range(100000):
	in1 = randint(min,max)
	in2 = randint(min,max)
	out = in1+in2
	if norm:
		in1 = (in1-min)/(max-min)
		in2 = (in2-min)/(max-min)
		out = (out-min)/(tmax-min)
	file.write(str(in1) + " " + str(in2) + " " + str(out) + "\n")
file.close()
