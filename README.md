# Combining Backpropagation with Equilibrium Propagation to improve an Actor-Critic RL framework 
This is code to reproduce our results on Acrobot-v1 from manuscript: "Combining Backpropagation with Equilibrium Propagation to improve an Actor-Critic RL framework":

To run epbp.py, go:

```
python epbp.py 
```

To run bp.py, go:

```
python bp.py 
```
*for this code, please install pytoch.
*This python code will create a directory "results_epbp" or "results_bp" to save the results (reward.npy).

After the training, these code plot the results. Also, you can check results whenever you want as follows:
go:

```
python show_reward.py 
```
*Please move "show_reward.py" to "results_acrbt" directory, and execute above.

If you want to change the task, please change:
Line 25 in epbp.py and bp.py for the task

Line 49 and 62 in epbp.py for the learning rates

Line 62 and 69 in epbp.py for the inputs and outputs

Line 325 and 326 in bp.py for the learning rates

Line 332 and 333 in bp.py for the outputs

Line 334 in bp.py for the inputs

*for bp.py, the specific learning rates are as follows:

Acrobot-v1: 2e-3 for actor and 1e-3 for critic

CartPole-v0: 2e-3 for actor and 1e-3 for critic

LunarLander-v2: 2e-3 for actor and 1e-4 for critic



