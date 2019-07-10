import tensorflow as tf
import pickle
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)
#from create_sentiment_featureset import create_featureset_and_labels

pickle_in = open('sentiment_set.pickle','rb')
train_x,train_y,test_x,test_y = pickle.load(pickle_in)

HIDDEN_LAYER_1 = 500
HIDDEN_LAYER_2 = 500
HIDDEN_LAYER_3 = 500

N_CLASSES = 2
BATCH_SIZE = 100

with tf.name_scope('input'): 
    texts = tf.placeholder('float',[None,len(train_x[0])],name="texts")
    labels = tf.placeholder('float',name="labels")

def fc_layer(input,size_out,name="fc",activation=None):
    with tf.name_scope(name):
        size_in = int(input.shape[1])
        w = tf.Variable(tf.random_normal([size_in,size_out]))
        b = tf.Variable(tf.random_normal([size_out]))
        wx_plus_b = tf.matmul(input,w) + b
        if activation:
            return activation(wx_plus_b)
        return wx_plus_b

def neural_network(x):
    fc1 = fc_layer(x,HIDDEN_LAYER_1,"fc1",activation=tf.nn.relu)
    fc2 = fc_layer(fc1,HIDDEN_LAYER_2,"fc2",activation=tf.nn.relu)
    fc3 = fc_layer(fc2,HIDDEN_LAYER_3,"fc3",activation=tf.nn.relu)
    out = fc_layer(fc3,N_CLASSES,name="output")
    return out

def train_neural_network(x):
    out = neural_network(x)
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=out),name="cost")
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(name="Adam").minimize(cost)

    NM_EPOCHS = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(NM_EPOCHS):
            epoch_loss=0
            
            i=0
            while i<len(train_x):
                start=i
                end = i+BATCH_SIZE

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _,c = sess.run([optimizer,cost],feed_dict={texts:batch_x,labels:batch_y})
                epoch_loss += c
                i += BATCH_SIZE
            print('Epoch ',epoch,' completed out of ',NM_EPOCHS,'loss :',epoch_loss)
            saver.save(sess,'./my-model/my-modell',global_step = epoch,write_meta_graph=True)

        correct = tf.equal(tf.argmax(out,1),tf.argmax(labels,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'),name="accuracy")
        print('Accuracy : ',accuracy.eval({texts:test_x,labels:test_y}))

train_neural_network(texts)
            
        
