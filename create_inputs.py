import random
import numpy as np
import pdb
import json
import pickle as pkl

class creator:

    def __init__(self, embed_size, user_embed, item_embed, user_history, user_key, item_key, content_embed):

        self.embed_size = embed_size
        self.user_embed = np.load(user_embed)
        self.item_embed = np.load(item_embed)
        self.user_history = json.load(open(user_history))
        self.user_key = pkl.load(open(user_key))
        self.user_key_rev = {v: k for k, v in self.user_key.iteritems()}
        self.item_key = pkl.load(open(item_key))
        self.item_key_rev = {v: k for k, v in self.item_key.iteritems()}
        self.content_embed = pkl.load(open(content_embed))

    def create_input(self, inp_size, n_negs):

        user_read = []
        user = []
        truth = []
        pos_neg_left = []
        pos_neg_right = []
        test_left = []
        test_right = []
        user_list = []
        user_read_list = []
        item_list_left = self.item_embed
        item_list_right = []
        for i in range(self.item_embed.shape[0]) :
            item_list_right.append(self.content_embed[str(self.item_key_rev[i])])

        for i in range(self.user_embed.shape[0]):
            total_hist = map(lambda x: self.content_embed[str(x)], self.user_history[self.user_key_rev[i]])
            total_hist2 = map(lambda x: self.item_embed[self.item_key[x]], self.user_history[self.user_key_rev[i]])
            size = len(total_hist)
            read_hist = []
            user_list.append(self.user_embed[i])
            test_left.append(total_hist2[-1])
            test_right.append(total_hist[-1])

            if size > inp_size - 1:
                n_pos = size - inp_size - 1
                read_hist = total_hist[:inp_size]
            else:
                n_pos = 1
                read_hist = []
                padding = np.zeros(self.embed_size)
                for j in range(inp_size - size + 2):
                    read_hist.append(padding)
                for j in range(size - 2):
                    read_hist.append(total_hist[j])
            
            user_read_list.append(read_hist)
            
            for j in range(n_pos):
                user_read.append(read_hist)
                user.append(self.user_embed[i])
                pos_neg_right.append(total_hist[size-2-j])
                pos_neg_left.append(total_hist2[size-2-j])
                truth.append(1)

            for j in range(n_pos*n_negs):
                user_read.append(read_hist)
                user.append(self.user_embed[i])
                pos_neg_right.append(random.choice(self.content_embed.values()))
                pos_neg_left.append(random.choice(self.item_embed))
                truth.append(0)

        return np.array(user), np.array(user_read), np.array(pos_neg_left), np.array(pos_neg_right), np.array(truth), np.array(test_left), np.array(test_right), np.array(user_list), np.array(item_list_left), np.array(item_list_right), np.array(user_read_list)
