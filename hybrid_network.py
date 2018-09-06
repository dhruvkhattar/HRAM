from __future__ import division
import numpy as np
import random
import pdb
from keras.layers import Layer, Input, merge, Dense, LSTM, Bidirectional, GRU, SimpleRNN
from keras.layers.merge import concatenate, dot, multiply
from keras.models import Model
from attention import AttentionWithContext
from keras.callbacks import ModelCheckpoint
from create_inputs import creator
from tqdm import tqdm
import os

class HybridModel:
    """
    This class is meant for the creation of the Hybrid Network.
    The main function in this class is the create_model function.
    """

    def __init__(self, embedding_size_useritem, embedding_size_article, history):
        """
        * Initialises variables to be used for history
        * Initialises article and item embedding shape
        """

        self.model = None
        self.history = history
        self.embedding_size_useritem = embedding_size_useritem
        self.embedding_size_article = embedding_size_article


    def create_model(self):
        """
        Creates the Hybrid Model.
        Consists of two components:
            * Left Component : Computes the user item interaction through matrix factorization (Typical Collaborative Filtering)
            * Right Component : Uses user history, item features, time etc (features)
              to dynamically model the user interests (Collaborative + Content Features)
        ===============================================================================
        Desc : Left Component
        ===============================================================================
        Desc : Right Component
        ===============================================================================
        """

        #Initialises input for left component
        user_embed = Input(shape=(self.embedding_size_useritem,))
        item_embed = Input(shape=(self.embedding_size_useritem,))

        #Initialises input for right component
        user_read = Input(shape=(self.history, self.embedding_size_article))
        user_case = Input(shape=(self.embedding_size_article, ))

        # Creates Layers for the left component
        concatenated_layer = concatenate([user_embed, item_embed])
        left_layer1 = Dense(128, activation='relu')(concatenated_layer)
        left_layer2 = Dense(64, activation='relu')(left_layer1)

        # Creates Layers for the right component
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(user_read)
        attention_layer = AttentionWithContext()(lstm_layer)

        right_layer_input = Dense(128, activation='relu')(user_case)

        elem_wise = multiply([attention_layer, right_layer_input])
        right_layer1 = Dense(64, activation='relu')(elem_wise)


        # Merges the left and right component
        merged_layer = concatenate([left_layer2, right_layer1])
        merged_layer1 = Dense(256, activation='relu')(merged_layer)
        merged_layer2 = Dense(128, activation='relu')(merged_layer1)
        merged_layer3 = Dense(64, activation='relu')(merged_layer2)
        output = Dense(1, activation='sigmoid')(merged_layer3)


        self.model = Model(inputs=[user_embed, item_embed] + [user_read] + [user_case], outputs=output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])



    def create_left_only(self):
        """
        Only Creates the Left Side of the Model.
        Specs remain the same
        """
        user_embed = Input(shape=(self.embedding_size_useritem,))
        item_embed = Input(shape=(self.embedding_size_useritem,))

        # Creates Layers for the left component
        concatenated_layer = concatenate([user_embed, item_embed])
        left_layer1 = Dense(128, activation='relu')(concatenated_layer)
        left_layer2 = Dense(64, activation='relu')(left_layer1)
        output = Dense(1, activation='sigmoid')(left_layer2)


        self.model = Model(inputs=[user_embed, item_embed], outputs=output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])


    def create_right_only(self):
        """
        Only creates the right side of the Model.
        Specs remain the same.
        """

        user_read = Input(shape=(self.history, self.embedding_size_article))
        user_case = Input(shape=(self.embedding_size_article, ))

        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(user_read)
        attention_layer = AttentionWithContext()(lstm_layer)

        right_layer_input = Dense(128, activation='relu')(user_case)

        elem_wise = multiply([attention_layer, right_layer_input])
        right_layer1 = Dense(64, activation='relu')(elem_wise)

        output = Dense(1, activation='sigmoid')(right_layer1)


        self.model = Model(inputs=[user_read] + [user_case], outputs=output)
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])



    def fit_model(self, inputs, output, pathname):
        if not os.path.exists("../weights/" + pathname):
            os.makedirs("../weights/" + pathname)
        filepath="../weights/"+pathname+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, output, validation_split=0.2, epochs=50, callbacks=callbacks_list, verbose=1) 

    def fit_model_left(self, inputs, output, pathname):
        if not os.path.exists("../weights/" + pathname):
            os.makedirs("../weights/" + pathname)
        filepath="../weights/"+pathname+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, output, validation_split=0.2, epochs=50, callbacks=callbacks_list, verbose=1) 

    def fit_model_right(self, inputs, output, pathname):
        if not os.path.exists("../weights/" + pathname):
            os.makedirs("../weights/" + pathname)
        filepath="../weights/"+pathname+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(inputs, output, validation_split=0.2, epochs=50, callbacks=callbacks_list, verbose=1) 

    def get_model_summary(self):
        print self.model.summary()

def train(model):
    c = creator(300, '../data/user_embed.npy', '../data/item_embed.npy', '../data/user_history.json', '../data/user_key.pkl', '../data/item_key.pkl', '../data/article_embed.pkl')
    user, user_read, pos_neg_left, pos_neg_right, truth, test_left, test_right, user_list, item_list_left, item_list_right, user_read_list = c.create_input(12, 5)
    model_test = HybridModel(128, 300, 12)
    if model == 'whole':
        model_test.create_model()
        model_test.get_model_summary()
        model_test.fit_model([user, pos_neg_left, user_read, pos_neg_right], truth, '12')
    elif model == 'left':
        model_test.create_left_only()
        model_test.get_model_summary()
        model_test.fit_model_left([user, pos_neg_left], truth, 'left/12')
    else: 
        model_test.create_right_only()
        model_test.get_model_summary()
        model_test.fit_model_right([user_read, pos_neg_right], truth, 'right/12')

def test(model, filename):
    c = creator(300, '../data/user_embed.npy', '../data/item_embed.npy', '../data/user_history.json', '../data/user_key.pkl', '../data/item_key.pkl', '../data/article_embed.pkl')
    user, user_read, pos_neg_left, pos_neg_right, truth, test_left, test_right, user_list, item_list_left, item_list_right, user_read_list = c.create_input(12, 5)
    model_test = HybridModel(128, 300, 12)
    if model == 'whole':
        model_test.create_model()
    elif model == 'left':
        model_test.create_left_only()
    else:
        model_test.create_right_only()
    model_test.get_model_summary()
    print user.shape, user_read.shape, pos_neg_left.shape, pos_neg_right.shape, truth.shape
    model_test.model.load_weights(filename)
    HR = []
    NDCG = []
    hr = [0]*10
    ndcg = [0]*10

    for i in tqdm(range(user_list.shape[0])):
    #for i in tqdm(range(100)):
        test_items_left = []
        test_items_right = []
        for k in range(99):
            test_items_left.append(item_list_left[random.randint(0, item_list_left.shape[0]-1)])
            test_items_right.append(item_list_right[random.randint(0, item_list_right.shape[0]-1)])
        test_items_left.append(test_left[i])
        test_items_right.append(test_right[i])
        if model == 'whole':
            out = model_test.model.predict([np.array([user_list[i]]*100), np.array(test_items_left), np.array([user_read_list[i]]*100), np.array(test_items_right)])
        elif model == 'right':
            out = model_test.model.predict([np.array([user_read_list[i]]*100), np.array(test_items_right)])
        else:
            out = model_test.model.predict([np.array([user_list[i]]*100), np.array(test_items_left)])
        sorted_items = [i[0] for i in sorted(enumerate(out), key=lambda x:x[1])]
        sorted_items.reverse()
        for k in range(10):
            rec = sorted_items[:k+1]
            if 99 in rec:
                hr[k] += 1
            for pos in range(k+1):
                if rec[pos] == 99:
                    ndcg[k] += 1 / np.log2(1+pos+1)
    
    for k in range(10):
        print k, 'hr',  hr[k], 'ndcg', ndcg[k]
        HR.append(float(hr[k]) / float(user_list.shape[0]))
        NDCG.append(float(ndcg[k]) / float(user_list.shape[0]))
        print k, 'HR',  HR[k], 'NDCG', NDCG[k]

def random_train():
    no_of_samples = 10
    # This is for the user embedding from siamese
    input1 = []
    for i in xrange(no_of_samples):
        input1.append(np.random.rand(300,))
    input1 = np.array(input1)
    
    # This is for the item embedding from siamese
    input2 = []
    no_of_samples = 10
    # This is for the user embedding from siamese
    input1 = []
    for i in xrange(no_of_samples):
        input1.append(np.random.rand(300,))
    input1 = np.array(input1)
    
    # This is for the item embedding from siamese
    input2 = []
    for i in xrange(no_of_samples):
        input2.append(np.random.rand(300,))
    input2 = np.array(input2)

    # This is for the user history used
    input3 = []
    for i in xrange(no_of_samples):
        input3.append(np.random.rand(12, 300))
    input3 = np.array(input3)

    # This is for the item positive/negative (same as the item from siamese)
    input4 = []
    for i in xrange(no_of_samples):
        input4.append(np.random.rand(300,))
    input4 = np.array(input4)

    model_test = HybridModel(300, 300, 12)
    output = np.random.randint(2, size=10)
    model_test.create_model()
    model_test.get_model_summary()
    model_test.fit_model([input1, input2, input3, input4], output, 'random')


if __name__ == "__main__":
    random_train()
    #train('whole')
