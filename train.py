import tensorflow as tf
import numpy as np


features = np.load("../../cs671_project_data/vecs_validation.npy")
#print(features.shape)

train_q1 = features[:, 0]
train_q2 = features[:, 1]
train_label = features[:, 2]

n_nodes_hl1 = 500
gamma=10

batch_size = 1
epsilon=0.0001234
x1 = tf.placeholder('float')
x2 = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data1, data2):

	hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([train_q1[0].shape[0], n_nodes_hl1])),
	                   'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

	l1_1 = tf.add(tf.matmul(data1, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1_2 = tf.add(tf.matmul(data2, hidden_1_layer['weights']), hidden_1_layer['biases'])

	l1_1 = tf.nn.tanh(l1_1)
	l1_2 = tf.nn.tanh(l1_2)
	
	r_q1 = tf.reduce_sum(l1_1, axis = 0)
	r_q2 = tf.reduce_sum(l1_2, axis = 0)
	
	r_q1 = tf.nn.tanh(r_q1)
	r_q2 = tf.nn.tanh(r_q2)

	normalize_a = tf.nn.l2_normalize(r_q1,0)        
	normalize_b = tf.nn.l2_normalize(r_q2,0)
	cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
	output =   tf.sigmoid(gamma*cos_similarity)
	return output

def train_neural_network(x1, x2):

	output_label = neural_network_model(x1, x2)
	
	cost = tf.reduce_mean(-1*y*tf.log(output_label+epsilon) -1*(1-y)*tf.log(1-output_label+epsilon))

	optimizer = tf.train.AdagradOptimizer(learning_rate=5).minimize(cost)

	hm_epochs = 50

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	
	ckpt_token = 1
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		#saver.restore(sess, "./model.ckpt")

		for epoch in range(hm_epochs):
			epoch_loss = 0

			i=0
			while i<len(train_q1):
				start = i
				end = i+batch_size
				epoch_q1 = np.array(train_q1[start]).transpose()
				epoch_q2 = np.array(train_q2[start]).transpose()
				epoch_label = train_label[start]

				#print(epoch_q1.shape, epoch_q2.shape, epoch_label.shape)
				# exit()
				_, c, _label = sess.run([optimizer, cost, output_label], feed_dict={x1:epoch_q1, x2: epoch_q2, y: epoch_label})
				print(_label, epoch_label, c)
				epoch_loss += c
				i += batch_size

				# if epoch_label == 1:
				# 	print("epoch label = " + str(epoch_label) + "; predicted label = " + str(_label))

				if i%45000 == 0:
					if ckpt_token == 1:
						save_path = saver.save(sess, "./model_checkpoint_pos.ckpt")
						file = open("train.ckpt_pos.log", 'w')
						file.write("Epoch:" + str(epoch) + " ckpt:" + str(i))
						file.close()
					else:
						save_path = saver.save(sess, "./model_checkpoint_neg.ckpt")
						file = open("train.ckpt_neg.log", 'w')
						file.write("Epoch:" + str(epoch) + " ckpt:" + str(i))
						file.close()
				
					ckpt_token = -ckpt_token

			

			print("****************************************************************************")
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
			print("****************************************************************************")


		save_path = saver.save(sess, "./model.ckpt") 
		# correct = tf.equal(tf.argmax(output_label,1), tf.argmax(y,1))

		# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


		# print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))


def feed_forward_nn():
	start = 0
	data1 = np.array(train_q1[start]).transpose()
	data2 = np.array(train_q2[start]).transpose()
	# print(data1.shape)
	W = np.random.randn(train_q1[0].shape[0], n_nodes_hl1)
	print("W.shape"),
	print(W.shape)
	b = np.random.randn(n_nodes_hl1)
	print("b.shape"),
	print(b.shape)

	l1_1 = np.matmul(data1, W)+ b
	print("l1_1.shape")
	print(l1_1.shape)
	l1_2 = np.matmul(data2, W)+ b
	print("l1_2.shape")
	print(l1_2.shape)
	
	l1_1 = np.tanh(l1_1)
	print("l1_1.shape")
	print(l1_1.shape)
	l1_2 = np.tanh(l1_2)
	print("l1_2.shape")
	print(l1_2.shape)

	r_q1 = np.sum(l1_1, axis = 0)
	print("r_q1.shape")
	print(r_q1.shape)
	r_q2 = np.sum(l1_2, axis = 0)
	print("r_q2.shape")
	print(r_q2.shape)
	
	r_q1 = np.tanh(r_q1)
	print("r_q1.shape")
	print(r_q1.shape)
	r_q2 = np.tanh(r_q2)
	print("r_q2.shape")
	print(r_q2.shape)

	normalize_a = r_q1/np.linalg.norm(r_q1,axis=0)
	print("normalize_a.shape")
	print(normalize_a.shape)
	normalize_b = r_q2/np.linalg.norm(r_q2,axis=0)
	print("normalize_b.shape")
	print(normalize_b.shape)

	cos_similarity=np.sum(np.multiply(normalize_a,normalize_b))
	
	output = cos_similarity
	print("cos_similarity.shape")
	print(cos_similarity.shape)
	return output

	
train_neural_network(x1, x2)

