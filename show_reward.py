# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:05:35 2020

@author: yoshi
"""

import numpy as np
#import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
arr_reward      = np.load('reward.npy',  allow_pickle=True)

length = len(arr_reward)
#print(arr_reward[4000:])
avg_reward   = []
each_reward = []
for i in range(length):
  if ((i + 1) % 10) == 0:
    avg_reward.append(np.mean(each_reward))
    each_reward = []   
  else:
    each_reward.append(arr_reward[i])

#np.save("reward_ep.npy", avg_reward)
print(np.max(avg_reward))
plt.plot(avg_reward)
#plt.plot(arr_reward)
plt.ylabel('Reward' , fontsize=20)
plt.xlabel('Episode (x 10)' , fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

