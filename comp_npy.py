import numpy as np

def comp_grad(f_ori, f_par):
	ori = np.load(f_ori)
	par = np.load(f_par)

	n_ori = np.sum(np.abs(ori) > 1.e+7)
	if n_ori > 0:
		print(n_ori)
		print('non-zero values in origin')
	n_par = np.sum(np.abs(par) > 1.e+7)
	if n_par > 0:
		print(n_par)
		print('non-zero values in compare')

	diff = np.sum(np.abs(ori - par) > 1.e+7)
	if diff > 0:
		print(diff)
		print('different voxels')
	else:
		print(diff)
		print('all voxels are the same')

	# for i in range(nrow):
	# 	for j in range(ncol):
	# 		for l in range(ndir):
	# 			diff = ori[i, j, l] - par[i, j, l]
	# 			if diff != 0.:
	# 				print(i, j, l)
	# else:
	# 	print('all pixels are the same')
