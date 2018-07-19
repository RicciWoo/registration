import numpy as np

def comp_grad(f_ori, f_par):
	ori = np.load(f_ori)
	par = np.load(f_par)

	nsli = ori.shape[0]
	nrow = ori.shape[1]
	ncol = ori.shape[2]
	ndir = ori.shape[3]

	diff = np.sum(np.abs(ori - par) > 1.e+7)
	if diff > 0:
		print('there are ' + diff + 'different voxels!')
	else:
		print('all voxels are the same!')

	# for k in range(nsli):
	# 	for i in range(nrow):
	# 		for j in range(ncol):
	# 			for l in range(ndir):
	# 				diff = ori[k, i, j, l] - par[k, i, j, l]
	# 				if diff != 0.:
	# 					print(k, i, j, l)
	# else:
	# 	print('all pixels are the same')

# A = np.ones((20, 20, 10))
# B = 2 * np.ones((20, 20, 10))
# s = np.sum(np.abs(A - B))