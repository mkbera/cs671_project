from gensim.models.keyedvectors import KeyedVectors
import nltk
import numpy as np


Word2Vec_word_vectors = KeyedVectors.load_word2vec_format('../../cs671_project_data/word2vec.bin', binary=True)


def getWord2Vector(s):
	words = nltk.word_tokenize(s)
	ret=[]
	# print(words)
	for w in words:
		if(w in Word2Vec_word_vectors.vocab):
			ret.append(Word2Vec_word_vectors[w])
		else:
			pass
			# print("ignoring", w)

	# print(len(ret))
	return ret


def getModWvec(wveclist, k=3):
	pad=Word2Vec_word_vectors['the']
	len_pad = len(pad)
	qmat = np.zeros((k*len_pad, len(wveclist)))
	
	for i in range(len(wveclist)):
		z=np.zeros((k*len_pad))
		z_index = 0

		# concatenating previous words
		for j in range((k-1)/2):
			index = i - (k-1)/2	 + j
			_vec = pad
			
			if index >=	0:
				_vec = wveclist[index]
			
			z[z_index: z_index+len_pad] = _vec
			z_index += len_pad

		# concatenating current word
		z[z_index: z_index+len_pad] = wveclist[i]
		z_index += len_pad

		# concatenating next word
		for j in range(k-1 - (k-1)/2):
			index = i + 1 + j
			_vec = pad

			if index < len(wveclist):
				_vec = wveclist[index]
			
			z[z_index: z_index+len_pad] = _vec
			z_index += len_pad

		qmat[:, i] = z

	# print(qmat.shape)
	return qmat			





