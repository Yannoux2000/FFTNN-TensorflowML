import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128
fft_length = 512

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit


x = tf.placeholder(tf.float32, [None, num_input])
y_ = tf.placeholder(tf.float32, [None, num_classes])

#Definition of same dense outputs
def DenseNetOutput(t_input, name="None"):
	with tf.name_scope("DenseOut_{}".format(name)):

		fc1 = tf.layers.dense(t_input, 1024)
		fc1 = tf.layers.dropout(fc1, rate=dropout)
		out = tf.nn.softmax(tf.layers.dense(fc1, num_classes))

	return out
#and same optimisation for better comparation
def OptimiseNetwork(t_logits,t_labels,name="None"):
	with tf.name_scope("Losses_{}".format(name)):

		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t_labels, logits=t_logits))

		train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

		correct_prediction = tf.equal(tf.argmax(t_logits,1), tf.argmax(t_labels,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		return train_step,cross_entropy,accuracy


with tf.name_scope("ConvNet"):

	# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
	x_reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])

	conv1 = tf.layers.conv2d(x_reshaped, 32, 5, activation=tf.nn.relu)
	conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

	conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
	conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

	cnn_flatten = tf.contrib.layers.flatten(conv2)

cnn_output = DenseNetOutput(cnn_flatten,name="ConvNet")
cnn_train,cnn_c_e,cnn_acc = OptimiseNetwork(cnn_output,y_,name="ConvNet")

with tf.variable_scope("rfftDenseNet"):

	x_fd = tf.spectral.rfft(x,fft_length=[fft_length])

	x_fd_re = tf.real(x_fd)
	x_fd_im = tf.imag(x_fd)

	# x_fd_arg = tf.angle(x_fd)
	# x_fd_abs = tf.abs(x_fd)

	h_conv1_re = tf.nn.relu(tf.layers.dense(x_fd_re, 512))
	h_conv2_re = tf.nn.relu(tf.layers.dense(h_conv1_re, 256))

	h_conv1_im = tf.nn.relu(tf.layers.dense(x_fd_im, 512))
	h_conv2_im = tf.nn.relu(tf.layers.dense(h_conv1_im, 256))

	zoutfft = tf.complex(h_conv2_re, h_conv2_im)

	#Complete analysis
	# h_conv1_arg = tf.nn.relu(tf.layers.dense(x_fd_arg, 512))
	# h_conv2_arg = tf.nn.relu(tf.layers.dense(h_conv1_arg, 256))

	# h_conv1_abs = tf.nn.relu(tf.layers.dense(x_fd_abs, 512))
	# h_conv2_abs = tf.nn.relu(tf.layers.dense(h_conv1_abs, 256))

	# W_re = tf.Variable(tf.random_normal([256]))
	# W_im = tf.Variable(tf.random_normal([256]))
	# W_arg = tf.Variable(tf.random_normal([256]))
	# W_abs = tf.Variable(tf.random_normal([256]))
	# B_complex = tf.Variable(tf.zeros([256]))

	# zoutfft_re = h_conv2_re * W_re + h_conv2_im * W_im + h_conv2_arg * W_arg + h_conv2_abs * W_abs + B_complex
	# zoutfft_im = h_conv2_re * W_re + h_conv2_im * W_im + h_conv2_arg * W_arg + h_conv2_abs * W_abs + B_complex

	# zoutfft = tf.complex(zoutfft_re, zoutfft_im)

	outfft = tf.spectral.irfft(zoutfft,fft_length=[fft_length])

fft_output = DenseNetOutput(outfft,name="FFTNet")
fft_train,fft_c_e,fft_acc = OptimiseNetwork(fft_output,y_,name="FFTNet")


#TensorBoard Stuff

#For cnn :
with tf.name_scope("Statistics"):
	times = tf.placeholder(tf.float32, [None])

	with tf.name_scope("CNN"):
		
		smry_cnn_c_e = tf.summary.scalar("Cross_Entropy", cnn_c_e)
		smry_cnn_acc = tf.summary.scalar("Accuracy", cnn_acc)
		smry_cnn_mean_t = tf.summary.scalar("Mean_Time", tf.reduce_mean(times))
		smry_cnn_max_t = tf.summary.scalar("Max_Time", tf.reduce_max(times))
		smry_cnn_min_t = tf.summary.scalar("Min_Time", tf.reduce_min(times))

		smry_cnn = tf.summary.merge([smry_cnn_c_e,smry_cnn_acc,smry_cnn_mean_t,smry_cnn_max_t,smry_cnn_min_t])

	with tf.name_scope("FFT"):
		smry_fft_c_e = tf.summary.scalar("Cross_Entropy", fft_c_e)
		smry_fft_acc = tf.summary.scalar("Accuracy", fft_acc)
		smry_fft_mean_t = tf.summary.scalar("Mean_Time", tf.reduce_mean(times))
		smry_fft_max_t = tf.summary.scalar("Max_Time", tf.reduce_max(times))
		smry_fft_min_t = tf.summary.scalar("Min_Time", tf.reduce_min(times))

		smry_fft = tf.summary.merge([smry_fft_c_e,smry_fft_acc,smry_fft_mean_t,smry_fft_max_t,smry_fft_min_t])


writer = tf.summary.FileWriter("./TB/cnn-vs-fft/2")

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

writer.add_graph(sess.graph)

print("  STARTING TESTS  ")
#Starting Tests

print("  FFT TURN  ")

#warmup
for epoch in range(100):

	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	sess.run(fft_train, feed_dict={x: batch_xs, y_: batch_ys})

runs = []
for epoch in range(num_steps):

	start_time = time.time()

	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	sess.run(fft_train, feed_dict={x: batch_xs, y_: batch_ys})

	runs.append(time.time() - start_time)

	if (epoch%10) == 0:
		s = sess.run(smry_fft,feed_dict={x: batch_xs, y_: batch_ys, times: runs})
		writer.add_summary(s,epoch)

print("  CNN TURN  ")

#warmup
for epoch in range(100):

	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	sess.run(cnn_train, feed_dict={x: batch_xs, y_: batch_ys})

runs = []
for epoch in range(num_steps):

	start_time = time.time()

	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	sess.run(cnn_train, feed_dict={x: batch_xs, y_: batch_ys})

	runs.append(time.time() - start_time)

	if (epoch%10) == 0:
		s = sess.run(smry_cnn,feed_dict={x: batch_xs, y_: batch_ys, times: runs})
		writer.add_summary(s,epoch)

#Adding final runs

s = sess.run(smry_fft,feed_dict={x: batch_xs, y_: batch_ys, times: runs})
writer.add_summary(s,epoch)

s = sess.run(smry_cnn,feed_dict={x: batch_xs, y_: batch_ys, times: runs})
writer.add_summary(s,epoch)
