import numpy as np

def comp_grad(f_ori, f_par):
	ori = np.load(f_ori)
	par = np.load(f_par)

	nsli = ori.shape[0]
	nrow = ori.shape[1]
	ncol = ori.shape[2]
	ndir = ori.shape[3]

	diff = np.sum(np.abs(ori - par) < 1.e+7)
	if diff > 0:
		print(diff)
		print('different voxels')
	else:
		print('all voxels are the same!')



# A = np.ones((20, 20, 10))
# B = 2 * np.ones((20, 20, 10))
# s = np.sum(np.abs(A - B))