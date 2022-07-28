# Combining Backpropagation with Equilibrium Propagation to improve an Actor-Critic RL framework 
This is code to reproduce our results on Acrobot-v1 from manuscript: "Combining Backpropagation with Equilibrium Propagation to improve an Actor-Critic RL framework":
The code for EP-BP (specifically, Actor) is based on "Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input" (https://github.com/ernoult/updatesEPgradientsBPTT)

To run epbp.py, go:

```
python epbp.py 
```

To run bp.py, go:

```
python bp.py 
```
*for this code, please install pytoch.
*These python codes will create a directory "results_epbp" or "results_bp" to save the results (reward.npy).

After the training, these codes plot the results. Also, you can check results when you want to do so as follows:
go:

```
python show_reward.py 
```
*Before executing above, please move "show_reward.py" to "results_epbp" or "results_bp" directory.

If you want to change the task, please change:
Line 25 in epbp.py and bp.py for the task

Line 49 and 62 in epbp.py for the learning rates

Line 62 and 69 in epbp.py for the inputs and outputs

Line 544 in epbp.py for the label

Line 300 and 301 in bp.py for the learning rates

Line 306 and 307 in bp.py for the outputs

Line 308 in bp.py for the inputs

*for bp.py, the specific learning rates (we carefully tune the learning rate on each task) are as follows:

Task             | learning rate for actor | learning rate for critic
-------------    | -------------           |-------------
CartPole-v0      | 2e-3                    | 1e-3
Acrobot-v1       | 2e-3                    | 1e-3
LunarLander-v2   | 2e-3                    | 1e-4 


