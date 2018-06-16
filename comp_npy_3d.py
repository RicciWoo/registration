import numpy as np

def comp_grad(f_ori, f_par):
	ori = np.load(f_ori)
	par = np.load(f_par)

	nsli = ori.shape[0]
	nrow = ori.shape[1]
	ncol = ori.shape[2]
	ndir = ori.shape[3]

	for k in range(nsli):
		for i in range(nrow):
			for j in range(ncol):
				for l in range(ndir):
					diff = ori[k, i, j, l] - par[k, i, j, l]
					if diff != 0.:
						print(k, i, j, l)
	else:
		print('all pixels are the same')
