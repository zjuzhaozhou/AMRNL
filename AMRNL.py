# This is the code of AAAI 2017 paper:
# Title: Community-based Question Answering via Asymmetric Multi-Faceted Ranking Network Learning
# Author: Zhou Zhao, Hanqing Lu, et al.
import tensorflow as tf
import cPickle as pickle
from scipy.sparse import coo_matrix

# read the data from pkl file
fq = open('Question.pkl', 'rb')
fa = open('Answer.pkl', 'rb')
fqa = open('QA_Relation.pkl', 'rb')
fgraph = open('Graph.pkl', 'rb')
que = pickle.load(fq)
ans = pickle.load(fa)
qa = pickle.load(fqa).astype(int)
graph = pickle.load(fgraph)
graph = graph.toarray().astype(float)

qa_size = qa.shape[0]
u_size = graph.shape[0]
q_size = que.shape[0]
a_size = ans.shape[0]
maxlens = que.shape[1]
que = que.reshape([q_size, maxlens, 1])
ans = ans.reshape([a_size, maxlens, 1])
constant_bias = 0.1
Lambda = 1

print "build phase"

# initialize the variable: input data for the neural network
triplet = tf.placeholder(tf.int32, [None, 5])
relation = tf.placeholder(tf.float32, [u_size, u_size])
question = tf.placeholder(tf.float32, [None, maxlens, 1])
answer = tf.placeholder(tf.float32, [None, maxlens, 1])

num_hidden = 128
# user authority
user = tf.Variable(tf.random_normal([u_size, num_hidden], mean=0, stddev=0.01))

cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

with tf.variable_scope('train'):
    val_q, state_q = tf.nn.dynamic_rnn(cell, question, dtype=tf.float32)
with tf.variable_scope('train', reuse=True):
    val_a, state_a = tf.nn.dynamic_rnn(cell, answer, dtype=tf.float32)

# transpose [1, 0, 2] means the original one is [0, 1, 2] and we swap the first 2 dimensions.
# swap the batch_size with sequence size.
val_q = tf.transpose(val_q, [1, 0, 2])
val_a = tf.transpose(val_a, [1, 0, 2])

# mean pooling layer.
last_q = tf.reduce_mean(val_q, 0)
last_a = tf.reduce_mean(val_a, 0)
c = tf.constant(constant_bias, shape=[1, 1])

Loss = []
# multi-faceted ranking function
for i in range(0, qa_size):
    q = tf.gather(last_q, triplet[i, 0:1])
    a1 = tf.gather(last_a, triplet[i, 1:2])
    u1 = tf.gather(user, triplet[i, 2:3])
    a2 = tf.gather(last_a, triplet[i, 3:4])
    u2 = tf.gather(user, triplet[i, 4:5])
    Loss.append(tf.maximum(c + tf.matmul(q, tf.transpose(a1))*tf.matmul(q, tf.transpose(u1))
                           - tf.matmul(q, tf.transpose(a2))*tf.matmul(q, tf.transpose(u2))
                           , tf.zeros([1, 1])))
Loss = tf.pack(Loss)

# user's expertise model
expert_norm = tf.reduce_sum(tf.square(tf.sub(user, tf.matmul(relation, user))))

# objective function
error_norm = tf.reduce_sum(Loss) + Lambda * expert_norm

# optimize the Loss function
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(error_norm)

# initialization
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

print "run phase"
# run phase
epoch = 100
for i in range(epoch):
    sess.run(minimize, {triplet: qa, question: que, answer: ans, relation: graph})
    print "______the training loss for epoch ", str(i), ":"
    print sess.run(error_norm, {triplet: qa, question: que, answer: ans, relation: graph})

sess.close()
