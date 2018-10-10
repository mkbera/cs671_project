from getVector import *
import csv
import numpy as np


def get_features(file="../../cs671_project_data/quora_train.tsv"):

	features = []

	with open(file, 'r') as csvin:
		csvin = csv.reader(csvin, delimiter='\t')
		for row in csvin:
			q1 = row[3]
			q2 = row[4]
			label = row[5]

			q1_wvec = getWord2Vector(q1)
			q2_wvec = getWord2Vector(q2)

			q1_modvec = getModWvec(q1_wvec)
			q2_modvec = getModWvec(q2_wvec)

			_feature = []
			_feature.append(q1_modvec)
			_feature.append(q2_modvec)
			_feature.append(int(label))

			features.append(_feature)

	return np.array(features)


features = get_features("../../cs671_project_data/quora_test.tsv")
np.save('../../cs671_project_data/vecs_test.npy', features)