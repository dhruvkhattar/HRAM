from __future__ import division
import numpy as np
import pdb
import json
import pickle as pkl
from tqdm import tqdm

class Data():

    def __init__(self, filename, low, high):

        fp = open(filename)
        all_user_history = json.load(fp)
        fp.close()
        self.item_history = {}
        self.user_history = {}
        items = []
        users = []
        
        user_history = {}
        for user in tqdm(all_user_history.keys()):
            if len(all_user_history[user]) <= high and len(all_user_history[user]) >= low:
                user_history[user] = all_user_history[user]
                items += user_history[user]
                users.append(user)

        items = list(set(items))
       
        users.sort()
        items.sort()

        user_key = {}
        item_key = {}

        ct = 0
        for each in items:
            item_key[each] = ct
            ct += 1
        
        ct = 0
        for each in users:
            user_key[each] = ct
            self.user_history[ct] = map(lambda x: item_key[x], user_history[each])
            ct += 1
        
        for user in tqdm(self.user_history.keys()):
            for item in self.user_history[user]:
                if not self.item_history.has_key(item):
                    self.item_history[item] = []
                self.item_history[item].append(user)
        
        self.user_sim = np.zeros((len(users), len(users)))
        self.item_sim = np.zeros((len(items), len(items)))
        
        pkl.dump(user_key, open('../data/user_key.pkl', 'w'))
        pkl.dump(item_key, open('../data/item_key.pkl', 'w'))
    
    
    def find_user_sim(self):

        for user1 in tqdm(self.user_history.keys()):
            for user2 in self.user_history.keys():
                intersection =  len(set(self.user_history[user1]) & set(self.user_history[user2]))
                union =  len(set(self.user_history[user1]) | set(self.user_history[user2]))
                self.user_sim[user1, user2] = intersection / union

        fp = open('../data/user_sim.npy', 'w')
        np.save(fp, self.user_sim)
        fp.close()

    def find_item_sim(self):

        for item1 in tqdm(self.item_history.keys()):
            for item2 in self.item_history.keys():
                intersection =  len(set(self.item_history[item1]) & set(self.item_history[item2]))
                union =  len(set(self.item_history[item1]) | set(self.item_history[item2]))
                self.item_sim[item1, item2] = intersection / union
        
        fp = open('../data/item_sim.npy', 'w')
        np.save(fp, self.item_sim)
        fp.close()

if __name__ == '__main__':
    
    d = Data('../data/user_history.json', 10, 15)
    d.find_user_sim()
    d.find_item_sim()
