from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pdb
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras import backend as K


class Siamese():

    def __init__(self):
        pass
    
    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        return K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


    def create_base_network(self, input_dim):
        '''Base network to be shared (eq. to feature extraction).
        '''
        seq = Sequential()
        seq.add(Dense(256, input_shape=(input_dim,), activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(128, activation='relu'))
        seq.add(Dropout(0.2))
        seq.add(Dense(128))
        return seq


    def compute_accuracy(self, predictions, labels):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        s = 0
        for i in range(labels.shape[0]):
            s += (labels[i] - predictions[i]) ** 2
        s /= labels.shape[0]
        return np.sqrt(s)


    def main(self, name):
        
        data = np.load('../data/'+name+'_sim.npy')

        total = data.shape[0]
        print(data.shape)
        train = (int)(0.7 * total)
        input_dim = 1
        epochs = 10
       
        tr_pairs = []
        tr_y = []
        for i in range(train):
            for j in range(train):
                tr_pairs += [[i,j]]
                tr_y += [data[i,j]]

        te_pairs = []
        te_y = []
        for i in xrange(train, total):
            for j in xrange(train, total):
                te_pairs += [[i,j]]
                te_y += [data[i,j]]

        tr_pairs = np.array(tr_pairs)
        tr_y = np.array(tr_y)
        te_pairs = np.array(te_pairs)
        te_y = np.array(te_y)

        print(tr_pairs.shape)
        print(te_pairs.shape)
        # network definition
        base_network = self.create_base_network(input_dim)

        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        # train
        rms = RMSprop()
        model.compile(loss='mean_squared_error', optimizer=rms)
        
        print(model.summary())
        
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=128,
                  epochs=epochs,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

        model.save('../data/' + name + '.h5')

        # compute final accuracy on training and test sets
        out = K.function([input_a, K.learning_phase()], [processed_a])
        test = []
        for i in range(total):
            test.append([i])

        embed = out([test,0])
        
        embed = np.array(embed)
        np.save(name+'_embed.npy', embed)
        pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        tr_acc = self.compute_accuracy(np.array(pred), np.array(tr_y))
        pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        te_acc = self.compute_accuracy(np.array(pred), np.array(te_y))

        print('RMSE on training set:',tr_acc)
        print('RMSE on test set:',te_acc)

if __name__ == '__main__':
    s = Siamese()
    s.main('item')
    s.main('user')
