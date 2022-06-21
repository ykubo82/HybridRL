# Author: Yoshimasa Kubo
# Date: 2021/09/09
# Updated: 2022/06/07
# Taks : Acrobot-v1

import gym
import argparse

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
import random

import os
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.cuda.FloatTensor)
FloatTensor = torch.cuda.FloatTensor

env = gym.make('Acrobot-v1')


print('observation space:', env.observation_space)
print('action space:', env.action_space)

directory = 'results_epbp'

f = None
exit_dir = os.path.isdir(directory)
if not exit_dir:
  os.mkdir(directory)    
  
parser = argparse.ArgumentParser(description='Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input')
parser.add_argument(
    '--episode',
    type=int,
    default=1000,
    metavar='ES',
    help='episode for training')
parser.add_argument(
    '--lr_a',
    nargs = '+',
    type=float,
    default=[1e-3, 1e-3],
    metavar='LR',
    help='learning rate (default: [1e-3, 1e-3])')
parser.add_argument(
    '--lr_c',
    type=float,
    default=1e-3,
    metavar='LR',
    help='learning rate (1e-3)')
parser.add_argument(
    '--size_tab_a',
    nargs = '+',
    type=int,
    default=[3, 256, 6],
    metavar='ST',
    help='tab of layer sizes (default: [10])')      
parser.add_argument(
    '--size_tab_c',
    nargs = '+',
    type=int,
    default=[1, 256, 6],
    metavar='ST',
    help='tab of layer sizes (default: [10])')   
parser.add_argument(
    '--batch_size',
    type=int,
    default=20,
    metavar='BT',
    help='mini-batch size for training') 
parser.add_argument(
    '--max_memory',
    type=int,
    default=1000,
    metavar='MX',
    help='maximum memory size for experience replay')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.95,
    metavar='G',
    help='discount factor for RL')
parser.add_argument(
    '--dt',
    type=float,
    default=0.2,
    metavar='DT',
    help='time discretization (default: 0.2)') 
parser.add_argument(
    '--T',
    type=int,
    default=150,
    metavar='T',
    help='number of time steps in the forward pass (default: 100)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=25,
    metavar='Kmax',
    help='number of time steps in the backward pass (default: 25)')  
parser.add_argument(
    '--beta',
    type=float,
    default=0.02,
    metavar='BETA',
    help='nudging parameter (default: 1)') 
parser.add_argument(
    '--delay',
    type=int,
    default=130,
    metavar='Delay',
    help='target delay (default: 80)') 
parser.add_argument(
    '--training-method',
    type=str,
    default='eqprop',
    metavar='TMETHOD',
    help='training method (default: eqprop)')
parser.add_argument(
    '--action',
    type=str,
    default='train',
    help='action to execute (default: train)')    
parser.add_argument(
    '--activation-function',
    type=str,
    default='hardsigm',
    metavar='ACTFUN',
    help='activation function (default: sigmoid)')
parser.add_argument(
    '--no-clamp',
    action='store_true',
    default=False,
    help='clamp neurons between 0 and 1 (default: True)')
parser.add_argument(
    '--discrete',
    action='store_true',
    default=False, 
    help='discrete-time dynamics (default: False)')
parser.add_argument(
    '--toymodel',
    action='store_true',
    default=False, 
    help='Implement fully connected toy model (default: False)')                                                    
parser.add_argument(
    '--device-label',
    type=int,
    default=0,
    help='selects cuda device (default 0, -1 to select )')
parser.add_argument(
    '--C_tab',
    nargs = '+',
    type=int,
    default=[],
    metavar='LR',
    help='channel tab (default: [])')
parser.add_argument(
    '--padding',
    type=int,
    default=0,
    metavar='P',
    help='padding (default: 0)')
parser.add_argument(
    '--Fconv',
    type=int,
    default=5,
    metavar='F',
    help='convolution filter size (default: 5)')
parser.add_argument(
    '--Fpool',
    type=int,
    default=2,
    metavar='Fp',
    help='pooling filter size (default: 2)')         
parser.add_argument(
    '--benchmark',
    action='store_true',
    default=False, 
    help='benchmark EP wrt BPTT (default: False)')

args = parser.parse_args()


if  args.activation_function == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))
    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))

elif args.activation_function == 'hardsigm':
    def rho(x):
       return x.clamp(min = 0).clamp(max = 1)

    def rhop(x):
        return (x >= 0) & (x <= 1)

elif args.activation_function == 'tanh':
    def rho(x):
        return torch.tanh(x)
    def rhop(x):
        return 1 - torch.tanh(x)**2 

def softmax(x):
  return F.softmax(x)

def hard_sigmoid(x):
  return (1+F.hardtanh(2*x-1))*0.5


# critic model trained by BP
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        input_size = args.size_tab_c[-1]
        hidden_size = args.size_tab_c[-2]
        output_size = args.size_tab_c[0]        
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = hard_sigmoid(self.l1(x))
        x = self.l2(x)
        return x

def learn(Qvals, values, critic, optim):
    loss  =  F.mse_loss(Qvals, values)

    optim.zero_grad()
    loss.backward()
    optim.step()
    torch.cuda.empty_cache()

# actor model trained by EP
class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.T = args.T
        self.Kmax = args.Kmax        
        self.dt = args.dt
        self.size_tab = args.size_tab_a
        self.lr_tab = args.lr_a
        self.ns = len(args.size_tab_a) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        if args.device_label >= 0:    
            device = torch.device("cuda:"+str(args.device_label))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False
        self.device = device
        self.beta = args.beta

        w = nn.ModuleList([])
                           
        for i in range(self.ns - 1):
            w.append(nn.Linear(args.size_tab_a[i + 1], args.size_tab_a[i], bias = True))
            w.append(None)
            
        w.append(nn.Linear(args.size_tab_a[-1], args.size_tab_a[-2]))                             
        self.w = w
        self = self.to(device)

    def stepper(self, data, s, target = None, beta = 0, reward=0, return_derivatives = False):
        dsdt = []
        dsdt.append(-s[0] + softmax(self.w[0](s[1])))
        if beta > 0:
            dsdt[0] = dsdt[0] + beta*reward*(target-s[0])

        for i in range(1, self.ns - 1):
            dsdt.append(-s[i] + rho(self.w[2*i](s[i + 1]) + torch.mm(s[i - 1], self.w[2*(i-1)].weight)))

        dsdt.append(-s[-1] + rho(self.w[-1](data) + torch.mm(s[-2], self.w[-3].weight)))

        for i in range(self.ns):
            s[i] = s[i] + self.dt*dsdt[i]	
                                     
        if return_derivatives:
           return s, dsdt
        else:
            return s
    
    def forward(self, data, s, seq = None, method = 'nograd', beta = 0, target = None, reward = None, delay=None, **kwargs):
        T = self.T
        Kmax = self.Kmax
        if len(kwargs) > 0:
            K = kwargs['K']
        else:
            K = Kmax
        if (method == 'withgrad'):
            for t in range(T):             
                if t == T - 1 - K:
                    for i in range(self.ns):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True    
                    data = data.detach()
                    data.requires_grad = True             
                s = self.stepper(data, s)
            return s
                
        elif (method == 'nograd'):
            delay_s = None
            if beta == 0:
                for t in range(T):                      
                    s = self.stepper(data, s)
                    if delay == t:
                      delay_s = s
                      
            else:
                for t in range(Kmax):                      
                    s = self.stepper(data, s, target, beta, reward)
                    
            return s, delay_s          
                    
        elif (method == 'nS'):
            s_tab = []
            for i in range(self.ns):
                s_tab.append([])
            
            criterion = nn.MSELoss(reduction = 'sum')
            for t in range(T):
                for i in range(self.ns):                 
                    s_tab[i].append(s[i])                    
                    s_tab[i][t].retain_grad()                      
                s = self.stepper(data, s)

            for i in range(self.ns):                 
                s_tab[i].append(s[i])                    
                s_tab[i][-1].retain_grad()                
            loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
            loss.backward()
            
            
            nS = []
            for i in range(self.ns):
                nS.append(torch.zeros(Kmax, 1, s[i].size(1), device = self.device))
               
            for t in range(Kmax):
                for i in range(self.ns):
                    ###############################nS COMPUTATION#####################################
                    if (t < i):
                        nS[i][t, :, :] = torch.zeros_like(nS[i][t, :, :])
                    else:    
                        nS[i][t, :, :] = (s_tab[i][T - t].grad).sum(0).unsqueeze(0)
                    ####################################################################################

                                      
               
            return s, nS     
            
        elif (method == 'dSDT'):

                DT = []

                for i in range(len(self.w)):
                    if self.w[i] is not None:
                        DT.append(torch.zeros(Kmax, self.w[i].weight.size(0), self.w[i].weight.size(1)))
                    else:
                        DT.append(None)        
                

                dS = []
                for i in range(self.ns):
                    dS.append(torch.zeros(Kmax, 1, self.size_tab[i], device = self.device))              
                
                    
                for t in range(Kmax):
                    s, dsdt = self.stepper(data, s, target, beta, return_derivatives = True)
                    ###############################dS COMPUTATION#####################################
                    for i in range(self.ns):
                        if (t < i):
                            dS[i][t, :, :] = torch.zeros_like(dS[i][t, :, :])
                        else:
                            dS[i][t, :, :] = -(1/(beta*s[i].size(0)))*dsdt[i].sum(0).unsqueeze(0)                        
                    ######################################################################################


                    #############DT COMPUTATION##################
                    gradw, _ = self.computeGradients(beta, data, s, seq)
                    for i in range(len(gradw)):
                        if gradw[i] is not None:
                            DT[i][t, :, :] = - gradw[i]
                    #####################################################
                                                       
                                                                             
        return s, dS, DT
        
        
    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns):
            s.append(torch.zeros(batch_size, self.size_tab[i], requires_grad = True))            
        return s
        
              
    def computeGradients(self, beta, data, s, seq):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)

                
        for i in range(self.ns - 1):
            gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(s[i], 0, 1), s[i + 1]) - torch.mm(torch.transpose(seq[i], 0, 1), s[i + 1]))) 
            gradw.append(None)            
            gradw_bias.append((1/(beta*batch_size))*(s[i] - seq[i]).sum(0))
            gradw_bias.append(None)                                                                                  
                                                                
        gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(s[-1] - seq[-1], 0, 1), data))
        gradw_bias.append((1/(beta*batch_size))*(s[-1] - seq[-1]).sum(0))
                                                                                                                                                                
        return  gradw, gradw_bias

  
    def updateWeights(self, beta, data, s, seq):
        gradw, gradw_bias = self.computeGradients(beta, data, s, seq)
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[i]
            if gradw_bias[i] is not None:
                self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw_bias[i]  
        

    def act(self, x, flg=False):
      action = np.random.choice(3, p=np.squeeze(x).cpu().detach().numpy())
      torch.cuda.empty_cache()
      return action, x 

  

  

## delete extra memory
## return ep_rewards, ep_actions, ep_observations, ep_observations_next, ep_done_or_not  
## 
def delete_memory(ep_rewards, ep_actions, ep_observations, ep_observations_next, ep_done_or_not, extra_size):
  extra_indx           = range(extra_size)
  ep_actions           = np.delete(ep_actions, extra_indx, axis=0)
  ep_rewards           = np.delete(ep_rewards, extra_indx, axis=0)
  ep_observations      = np.delete(ep_observations, extra_indx, axis=0)
  ep_observations_next = np.delete(ep_observations_next, extra_indx, axis=0)
  ep_done_or_not = np.delete(ep_done_or_not, extra_indx, axis=0) 
     
  return ep_rewards, ep_actions, ep_observations, ep_observations_next, ep_done_or_not  

## store experience memory
## return ep_rewards, ep_actions, ep_observations, ep_observations_next, ep_done_or_not  
## 
def store_memory(ep_rewards, ep_actions, ep_observations, ep_observations_next, ep_done_or_not,  rewards, ys, observations, observations_next, done_or_not, episode):
  if episode == 0:    
    ep_rewards           = np.vstack(rewards)         
    ep_actions           = np.vstack(ys.cpu().numpy())     
    ep_observations      = np.vstack(observations)
    ep_observations_next = np.vstack(observations_next)    
    ep_done_or_not       = np.vstack(done_or_not)     
  else:
    ep_rewards           = np.vstack((ep_rewards, np.expand_dims(rewards, axis=1)))
    ep_actions           = np.vstack((ep_actions, ys.cpu().numpy()))

    observations         = np.vstack(observations)
    observations_next    = np.vstack(observations_next)

    ep_observations      = np.vstack((ep_observations, observations))
    ep_observations_next = np.vstack((ep_observations_next, observations_next))
    ep_done_or_not       = np.vstack((ep_done_or_not, np.expand_dims(done_or_not, axis=1)))       
  return ep_rewards, ep_actions, ep_observations, ep_observations_next, ep_done_or_not  


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

## train the model  
## return training and testing accuraices
##
def train_model(episode, actor, optim, critic, gamma, delay, batch_size, max_memory, render=False):
 
 
  ep_observations       = []
  ep_observations_next  = []
  ep_rewards            = []
  ep_actions            = []
  ep_done_or_not        = []
  reward_all            = []
  
  for i in range(episode):
    done = False    
    observation = env.reset()

    observations    = []
    observations_next = []    
       
    actions         = []
    rewards         = []
    ys              = FloatTensor([])
    done_or_not     = []

    reward_sum   = 0
    while not done:
      if render:
        env.render()        

      observation = torch.unsqueeze(torch.from_numpy(observation),0)
      observations.append(observation)

      s = actor.initHidden(observation.float().cuda().size(0))
      if actor.cuda:
        for j in range(actor.ns):
          s[j] = s[j].to(actor.device)

      with torch.no_grad():
        s, _ = actor.forward(observation.float().cuda(), s)
        action, _ = actor.act(s[0])        
          
      observation_next, reward, done, info = env.step(action)
      reward_sum += reward
      
      # fake target
      y =  FloatTensor([[1, 0, 0]] if action == 0 else [[0, 1, 0]] if action == 1 else [[0, 0, 1]])      
      ys = torch.cat([ys, y])
      
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
          ep_rewards, ep_actions, ep_observations, ep_observations_next, ep_done_or_not  = delete_memory(ep_rewards, ep_actions, ep_observations,
                                                                                                         ep_observations_next, 
                                                                                                         ep_done_or_not, extra_size)      

        batch_size_l2 = np.shape(ep_observations)[0]            
        batch_data_index = random.sample(range(batch_size_l2),batch_size)   

        # critic with BP    
        values = critic(FloatTensor(ep_observations[batch_data_index]).float().cuda())
        q_act = critic(FloatTensor(ep_observations_next[batch_data_index]).float().cuda())
        done_  = ep_done_or_not[batch_data_index]*1*(-1) + 1
        Qvals  = np.zeros_like(q_act.cpu().detach().numpy())
        Qval   = q_act.cpu().detach().numpy()        
        Qval   = Qval*done_
        
        Qvals = ep_rewards[batch_data_index] + gamma * Qval   
    
        Qvals = torch.Tensor(Qvals)

        # advantage values + normalization
        adv_dis_epr = Qvals - values           
        adv_dis_epr = (adv_dis_epr - torch.mean(adv_dis_epr)) / (torch.std(adv_dis_epr) + 1e-10) 
            
        # bp learning for actor
        learn(Qvals, values, critic, optim) 

        s = actor.initHidden(FloatTensor(ep_observations[batch_data_index]).float().cuda().size(0))
        if actor.cuda:
          for j in range(actor.ns):
            s[j] = s[j].to(actor.device)
  
        with torch.no_grad():
          _, s = actor.forward(FloatTensor(ep_observations[batch_data_index]).float().cuda(), s, delay=delay)
             
          ###################################* EQPROP *############################################
          seq = []
          for j in range(len(s)):
            seq.append(s[j].clone())
          s, _ = actor.forward(FloatTensor(ep_observations[batch_data_index]).float().cuda(), s, target = torch.tensor(ep_actions[batch_data_index]), beta = actor.beta, reward=adv_dis_epr, method = 'nograd')    
          actor.updateWeights(actor.beta, FloatTensor(ep_observations[batch_data_index]).float().cuda(), s, seq)
          #########################################################################################


    ## store experiences to memory
    ep_rewards, ep_actions, ep_observations, ep_observations_next, ep_done_or_not  = store_memory(ep_rewards, ep_actions, 
                                                                                                  ep_observations, ep_observations_next, 
                                                                                                  ep_done_or_not,  rewards, ys, 
                                                                                                  observations, observations_next, 
                                                                                                  done_or_not, i)
    ## append each episode reward
    reward_all.append(reward_sum)       
    np.save(directory + '/reward.npy', reward_all)
  
    print("episode :" + str(i))
    print(reward_sum)
    torch.cuda.empty_cache()
  return reward_all

def run():

  # EP model for Actor
  actor         = Actor(args)
  
  # backprop model for Critic
  critic         = Critic(args)
  optim          = torch.optim.Adam(critic.parameters(), lr=args.lr_c)
  use_cuda       = torch.cuda.is_available()
  if use_cuda:
    critic.cuda()

  return train_model(args.episode, actor, optim, critic, args.gamma, args.delay, args.batch_size, args.max_memory)

# run the code
rewards = run()

# show rewards
show_rewards(rewards)
