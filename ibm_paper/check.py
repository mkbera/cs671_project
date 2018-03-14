# f = "temptest.txt"
n_line = 1
while True:
	a = raw_input();
	if a == '':
		break;
	label = int(a[1])
	pred = int(a[4])
	# print(a[4])
	# exit()
	if label != pred:
		print(n_line, label, pred)
	n_line += 1
