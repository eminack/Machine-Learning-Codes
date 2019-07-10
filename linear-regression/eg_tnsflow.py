import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

HIDDEN_LAYER_1 = 500
HIDDEN_LAYER_2 = 250

N_CLASSES = 10
BATCH_SIZE = 1000

with tf.name_scope('input'):
    images = tf.placeholder('float',[None,784])
    labels = tf.placeholder('float')

def fc_layer(input,size_out,name="fc",activation=None):
    with tf.name_scope(name):
        size_in = int(input.shape[1])
        w = tf.Variable(tf.random_normal([size_in,size_out]))
        b = tf.Variable(tf.random_normal([size_out]))
        wx_plus_b = tf.matmul(input,w) + b
        if activation:
            return activation(wx_plus_b)
        return wx_plus_b


def train_neural_network(x):
    fc1 = fc_layer(x,HIDDEN_LAYER_1,"fc1",activation=tf.nn.relu)
    fc2 = fc_layer(fc1,HIDDEN_LAYER_2,"fc2",activation=tf.nn.relu)
    out = fc_layer(fc2,N_CLASSES,name="output")
    
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=out))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    NM_EPOCHS = 60

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(NM_EPOCHS):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/BATCH_SIZE)):
                epoch_x,epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _,c = sess.run([optimizer,cost],feed_dict={images:epoch_x,labels:epoch_y})
                epoch_loss += c
            print('Epoch ',epoch,' completed out of ',NM_EPOCHS,'loss :',epoch_loss)
        
        correct = tf.equal(tf.argmax(out,1),tf.argmax(labels,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy : ',accuracy.eval({images:mnist.test.images,labels:mnist.test.labels}))

train_neural_network(images)
            
        
