import numpy as np

def comp_grad(f_ori, f_par):
	ori = np.load(f_ori)
	par = np.load(f_par)

	nrow = ori.shape[0]
	ncol = ori.shape[1]
	ndir = ori.shape[2]

	for i in range(nrow):
		for j in range(ncol):
			for l in range(ndir):
				diff = ori[i, j, l] - par[i, j, l]
				if diff != 0.:
					print(i, j, l)
	else:
		print('all pixels are the same')
