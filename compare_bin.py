def compare_bin(fname1, fname2):
	f1 = open(fname1, 'rb')
	try:
		byte1 = f1.read(1)
		while byte1 != '':
			byte1 = f1.read(1)
	finally:
		f1.close
	f2 = open(fname2, 'rb')
	try:
		byte2 = f2.read(1)
		while byte2 != '':
			byte2 = f2.read(1)
	finally:
		f2.close
	