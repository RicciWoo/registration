import numpy as np

def comp_pdf(f_ori, f_par):
	ori = np.load(f_ori)
	par = np.load(f_par)

	nsli = ori.shape[0]
	nrow = ori.shape[1]
	ncol = ori.shape[2]

	for i in range(nsli):
		for j in range(nrow):
			for k in range(ncol):
				diff = abs(ori[i, j, k] - par[i, j, k])
				if diff < 1.e-7:
					print(i, j, k, diff)
	else:
		print('all pixels are the same')
