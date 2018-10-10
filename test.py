import tensorflow as tf
import numpy as np


features = np.load("../../cs671_project_data/vecs_test.npy")

test_q1 = features[:, 0]
test_q2 = features[:, 1]
test_label = features[:, 2]

n_nodes_hl1 = 500
gamma=10
batch_size = 1

x1 = tf.placeholder('float')
x2 = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data1, data2):

	hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([test_q1[0].shape[0], n_nodes_hl1])),
	                   'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

	l1_1 = tf.add(tf.matmul(data1, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1_2 = tf.add(tf.matmul(data2, hidden_1_layer['weights']), hidden_1_layer['biases'])

############################
	l1_1 = tf.nn.tanh(l1_1)
	l1_2 = tf.nn.tanh(l1_2)
############################
	
	r_q1 = tf.reduce_sum(l1_1, axis = 0)
	r_q2 = tf.reduce_sum(l1_2, axis = 0)
	
	r_q1 = tf.nn.tanh(r_q1)
	r_q2 = tf.nn.tanh(r_q2)

	normalize_a = tf.nn.l2_normalize(r_q1,0)        
	normalize_b = tf.nn.l2_normalize(r_q2,0)
	cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
	
	
	output = cos_similarity
	output =   tf.sigmoid(gamma*cos_similarity)
	return output
	# euclid =   tf.reduce_sum(tf.multiply(r_q1, r_q1)) + tf.reduce_sum(tf.multiply(r_q2, r_q2)) -2 * tf.reduce_sum(tf.multiply(r_q1, r_q2))
	# output = tf.exp(-1*euclid)



	return output



def test_neural_network(x1, x2):

	output_label = neural_network_model(x1, x2)

	# cost = tf.losses.mean_squared_error(y, output_label)

	# optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 1

	

	with tf.Session() as sess:
		# sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(sess, "./model.ckpt")
	
		zero_disc = 0
		one_disc = 0
		count = 0
		tp=0
		fp=0
		tn=0
		fn=0
		i=0
		while i<len(test_q1):
			start = i
			end = i+batch_size
			epoch_q1 = np.array(test_q1[start]).transpose()
			epoch_q2 = np.array(test_q2[start]).transpose()
			epoch_label = test_label[start]

			prediction = sess.run([output_label], feed_dict={x1:epoch_q1, x2: epoch_q2})
			prediction = np.squeeze(prediction)

			i += batch_size
			if epoch_label == 1:
				if prediction >= 0.7:
					prediction = 1
					count += 1
					tp += 1
				else:
					prediction = 0
					fn += 1
					one_disc += 1
			else:
				if prediction < 0.7:
					count += 1
					prediction = 0
					tn += 1
				else:
					prediction = 1
					fp += 1
					zero_disc += 1 

			print(epoch_label, prediction)




		print('Overall_Accuracy:', count, len(test_q1),(count*1.0)/len(test_q1))
		print('Precision: ', (tp*1.0)/(tp+fp))
		print('Recall: ', (tp*1.0)/(tp+fn))
		print('Zero tha par 1 predict kiya:', zero_disc)
		print('One tha par 0 predict kiya:', one_disc)


test_neural_network(x1, x2)
