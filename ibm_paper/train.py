import tensorflow as tf
import numpy as np
from preprocess import *


features = get_features("../../cs671_project_data/quora_train.tsv")

train_q1 = features[:, 0]
train_q2 = features[:, 1]
train_label = features[:, 2]

n_nodes_hl1 = 500

batch_size = 1

x1 = tf.placeholder('float')
x2 = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data1, data2):

	hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([train_q1[0].shape[0], n_nodes_hl1])),
	                   'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

	
	

	# output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
	#                    'biases' : tf.Variable(tf.random_normal([n_classes]))}


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
	return output



def train_neural_network(x1, x2):

	output_label = neural_network_model(x1, x2)

	cost = tf.losses.mean_squared_error(y, output_label)

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		for epoch in range(hm_epochs):
			epoch_loss = 0

			i=0
			while i<len(train_q1):
				start = i
				end = i+batch_size
				epoch_q1 = np.array(train_q1[start]).transpose()
				epoch_q2 = np.array(train_q2[start]).transpose()
				epoch_label = train_label[start]

				# print(epoch_q1.shape, epoch_q2.shape, epoch_label.shape)
				# exit()
				_, c = sess.run([optimizer, cost], feed_dict={x1:epoch_q1, x2: epoch_q2, y: epoch_label})

				epoch_loss += c
				i += batch_size

				if i%90 == 0:
					save_path = saver.save(sess, "./model_checkpoint.ckpt")
					file = open("train.ckpt.log", 'w')
					file.write("Epoch:" + str(epoch) + " ckpt:" + str(i))

			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)


		save_path = saver.save(sess, "./model.ckpt") 
		# correct = tf.equal(tf.argmax(output_label,1), tf.argmax(y,1))

		# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


		# print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))



train_neural_network(x1, x2)