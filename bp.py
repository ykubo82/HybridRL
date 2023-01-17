# Author: Yoshimasa Kubo
# Date: 2021/09/09
# Updated: 
# dataset : cart pole for BP
import gym

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import random


import os
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.cuda.FloatTensor)


env = gym.make('Acrobot-v1')

FloatTensor = torch.cuda.FloatTensor


print('observation space:', env.observation_space)
print('action space:', env.action_space)

directory = 'results_bp'

f = None
exit_dir = os.path.isdir(directory)
if not exit_dir:
  os.mkdir(directory) 


def linear(x):
  return x

def hard_sigmoid(x):
  # this is better for cartpole, acrobot, and lunerlander on BP
  # but if you want to try another hard sigmoid that we explain in the paper
  # you can try line 50
  return (1+F.hardtanh(2*x-1))*0.5
  
  # return x.clamp(min = 0).clamp(max = 1)     

def dr_hard_sigmoid(x):
  return (x >= 0) & (x <= 1)

  
def act_bp(x, flg=False):
  action = 0 if np.random.random() < x.data[0][0] else 1

  return action, x 

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = hard_sigmoid(self.l1(x))
        x = self.l2(x)
        return x
   

def act_bp_batch(x):
  return x.max(1)[0].detach()

def critic_learn(Qvals, values, critic, optim_critic):
    loss  =  F.mse_loss(Qvals, values)
    optim_critic.zero_grad()

    loss.backward()
    optim_critic.step()
    torch.cuda.empty_cache()

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x_temp   = self.l1(x)
        x1       = hard_sigmoid(x_temp)
        x_1_temp = self.l2(x1)
        x2       = F.softmax(x_1_temp)
        return x2, x1, x_temp, x_1_temp     

    def act(self, x, flg=False):
      action = np.random.choice(3, p=np.squeeze(x).cpu().detach().numpy())
      torch.cuda.empty_cache()
      return action, x 


## back prop way to train actor 
##
def actor_learn_own(input_, hidden_act, hidden_act_1, hidden_act_2, y, action_pred, adv, actor, lr=0.01):
   dlogprob        = y - action_pred
   batch_size = np.shape(y)[0]
   dlogprob_reward = dlogprob * adv 

   dW2 = torch.mm(hidden_act.T, dlogprob_reward) 

   dZ1 = torch.mm(actor.l2.weight.T, dlogprob_reward.T)
   grad_hiddden_act_1 = dr_hard_sigmoid(hidden_act_1)
   dZ1 *= grad_hiddden_act_1.T
   dW1 = torch.mm(input_.T, dZ1.T)
   with torch.no_grad():

     actor.l2.weight += lr * (dW2.T/float(batch_size))
     actor.l1.weight += lr * (dW1.T/float(batch_size))

## show rewards vs epoch
##     
def show_rewards(rewards):
  length = len(rewards)

  avg_reward   = []
  each_reward = []
  for i in range(length):
    if ((i + 1) % 10) == 0:
      avg_reward.append(np.mean(each_reward))
      each_reward = []   
    else:
      each_reward.append(rewards[i])
  
  plt.plot(avg_reward)
  plt.ylabel('Reward' , fontsize=18)
  plt.xlabel('Episode (x 10)' , fontsize=18)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.show()
  

def delete_list_pytorch(list_, indx):
    max_val = indx[-1] 
    stay_indx = range(len(list_))[max_val+1:]
    return list_[stay_indx]
    

## delete extra memory
## return ep_dlogp, ep_rewards, ep_observations, ep_disc_epr, ep_free_acts
## 
def delete_memory(ep_rewards, ep_actions, ep_probs, ep_hidden_acts, ep_hidden_acts_1, ep_hidden_acts_2, ep_observations, ep_observations_next, ep_done_or_not, extra_size):
  extra_indx           = range(extra_size)
  ep_actions           = np.delete(ep_actions, extra_indx, axis=0)
  ep_probs             = delete_list_pytorch(ep_probs, extra_indx)
  ep_rewards           = np.delete(ep_rewards, extra_indx, axis=0)
  ep_hidden_acts       = np.delete(ep_hidden_acts, extra_indx, axis=0)  
  ep_hidden_acts_1     = np.delete(ep_hidden_acts_1, extra_indx, axis=0)  
  ep_hidden_acts_2     = np.delete(ep_hidden_acts_2, extra_indx, axis=0)  
  ep_observations      = np.delete(ep_observations, extra_indx, axis=0)
  
  ep_observations_next = np.delete(ep_observations_next, extra_indx, axis=0)
  ep_done_or_not = np.delete(ep_done_or_not, extra_indx, axis=0) 
     
  return ep_rewards, ep_actions, ep_probs, ep_hidden_acts, ep_hidden_acts_1, ep_hidden_acts_2, ep_observations, ep_observations_next, ep_done_or_not  

## train the model  
## return training and testing accuraices
##
def train_model(episode, gam, actor, critic, optim_actor, optim_critic, lr_critic, lr_actor, batch_size, max_memory, render=False):
               
 
  ep_observations       = []
  ep_observations_next  = []
  ep_rewards            = []
  ep_actions            = []
  reward_all            = []
  ep_done_or_not        = []
  ep_probs              = []
  ep_hidden_acts        = []
  ep_hidden_acts_1      = []
  ep_hidden_acts_2      = []
  for i in range(episode):
    done = False    
    observation = env.reset()

    observations      = []
    observations_next = []    
   
    
    actions         = []
    rewards         = []
    ys              = FloatTensor([])
    probs           = FloatTensor([])
    done_or_not     = []
    hidden_acts     = FloatTensor([])
    hidden_acts_1   = FloatTensor([])    
    hidden_acts_2   = FloatTensor([])        
    reward_sum      = 0

    while not done:
      if render:
        env.render()        

      observation = torch.unsqueeze(torch.from_numpy(observation),0)
      observations.append(observation)

      action_prob, hidden_act,  x_temp, x_1_temp  = actor(observation.float().cuda())
      action, _    = actor.act(action_prob)
          
      observation_next, reward, done, info = env.step(action)
      reward_sum += reward

      # fake target
      y =  FloatTensor([[1, 0, 0]] if action == 0 else [[0, 1, 0]] if action == 1 else [[0, 0, 1]])      
      ys = torch.cat([ys, y])
      probs = torch.cat([probs, action_prob])
      hidden_acts = torch.cat([hidden_acts, hidden_act])
      hidden_acts_1 = torch.cat([hidden_acts_1, x_temp])      
      hidden_acts_2 = torch.cat([hidden_acts_2, x_1_temp])            
      
      observation = observation_next
      
      rewards.append(reward)
      actions.append(action)
      observations_next.append(observation_next)
      done_or_not.append(done)

      # experience replay
      batch_size_l = np.shape(ep_observations)[0]    
      if batch_size_l >= batch_size:
        # delete memory
        if max_memory < batch_size_l:
          extra_size = batch_size_l - max_memory
          ep_rewards, ep_actions, ep_probs, ep_hidden_acts, ep_hidden_acts_1, ep_hidden_acts_2, ep_observations, ep_observations_next, ep_done_or_not  = delete_memory(ep_rewards, ep_actions, ep_probs,  ep_hidden_acts, ep_hidden_acts_1, ep_hidden_acts_2, ep_observations, ep_observations_next, ep_done_or_not, extra_size)      

        batch_size_l2 = np.shape(ep_observations)[0]            
        batch_data_index = random.sample(range(batch_size_l2),batch_size)   

               
        # critic with BP    
        values = critic(FloatTensor(ep_observations[batch_data_index]).float().cuda())
        q_act  = critic(FloatTensor(ep_observations_next[batch_data_index]).float().cuda())
        done_  = ep_done_or_not[batch_data_index]*1*(-1) + 1
        Qvals  = np.zeros_like(q_act.cpu().detach().numpy())
        Qval   = q_act.cpu().detach().numpy()        
        Qval   = Qval*done_

        
        Qvals = ep_rewards[batch_data_index] + gam * Qval   
        Qvals = torch.Tensor(Qvals)

        adv_dis_epr = Qvals - values           
        adv_dis_epr = (adv_dis_epr - torch.mean(adv_dis_epr)) / (torch.std(adv_dis_epr) + 1e-10) 
       
        # bp learning for critic
        critic_learn(Qvals, values, critic, optim_critic) 
        
        # bp learning for actor
        actor_learn_own(FloatTensor(ep_observations[batch_data_index]), FloatTensor(ep_hidden_acts[batch_data_index]), FloatTensor(ep_hidden_acts_1[batch_data_index]), FloatTensor(ep_hidden_acts_2[batch_data_index]), FloatTensor(ep_actions[batch_data_index]), ep_probs[batch_data_index].clone(), adv_dis_epr, actor, lr_actor)        
        

    if i == 0:    
      ep_rewards           = np.vstack(rewards)         
      ep_actions           = np.vstack(ys.cpu().numpy())     
      ep_observations      = np.vstack(observations)
      ep_observations_next = np.vstack(observations_next)    
      ep_done_or_not       = np.vstack(done_or_not)   
      ep_probs             = probs
      ep_hidden_acts       = np.vstack(hidden_acts.detach().cpu().numpy())   
      ep_hidden_acts_1     = np.vstack(hidden_acts_1.detach().cpu().numpy())         
      ep_hidden_acts_2       = np.vstack(hidden_acts_2.detach().cpu().numpy())               
    else:
      ep_rewards           = np.vstack((ep_rewards, np.expand_dims(rewards, axis=1)))
      ep_actions           = np.vstack((ep_actions, ys.cpu().numpy()))

      observations         = np.vstack(observations)
      observations_next    = np.vstack(observations_next)

      ep_observations      = np.vstack((ep_observations, observations))
      ep_observations_next = np.vstack((ep_observations_next, observations_next))
      ep_done_or_not       = np.vstack((ep_done_or_not, np.expand_dims(done_or_not, axis=1)))            
      ep_probs             = torch.cat([ep_probs, probs])
      ep_hidden_acts       = np.vstack((ep_hidden_acts, hidden_acts.detach().cpu().numpy()))            
      ep_hidden_acts_1     = np.vstack((ep_hidden_acts_1, hidden_acts_1.detach().cpu().numpy()))
      ep_hidden_acts_2      = np.vstack((ep_hidden_acts_2, hidden_acts_2.detach().cpu().numpy()))               
    
    reward_all.append(reward_sum)       
    np.save(directory + '/reward.npy', reward_all)

    print("episode :" + str(i))
    print(reward_sum)
    torch.cuda.empty_cache()
  return reward_all

def run():

  ## setting up the hyper parameters 
  episode            = 1000             # training episode
  gam                = 0.95             # gamma for RL 

  lr_actor           = 2e-3             # learning rate for actor
  lr_critic          = 1e-3             # learning rate for critic  
  
  hidden_size_critic = 256              # hidden size for the network  
  hidden_size_actor  = 256              # hidden size for the network
  
  output_size_actor  = 3                # output size for the network
  output_size_critic = 1                # output size for the network  
  input_size         = 6                # input size for the network 
  
  batch_size         = 20
  max_memory         = 1000


  # backprop model for actor
  actor         = Actor(input_size, hidden_size_actor, output_size_actor)
  optim_actor   = torch.optim.Adam(actor.parameters(), lr=lr_actor)
  
  # backprop model for critic
  critic         = Critic(input_size, hidden_size_critic, output_size_critic)
  optim_critic   = torch.optim.Adam(critic.parameters(), lr=lr_critic)
  use_cuda       = torch.cuda.is_available()
  if use_cuda:
    critic.cuda()
    actor.cuda()

  return train_model(episode, gam, actor, critic, optim_actor, optim_critic, lr_critic, lr_actor, batch_size, max_memory)

# run the code
rewards = run()

# show rewards
show_rewards(rewards)
