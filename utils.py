import os
import math
import random
from collections import namedtuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

import torch
import torch.nn.functional as F
import torchvision.transforms as T

def save_checkpoint(model, optim, name, extra=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'extra': extra,
    }, "./checkpoints/%s.tar" % name)
    '''torch.save(model.state_dict(), "./checkpoints/%s_param_state.pt" % name)
    torch.save(optim.state_dict(), "./checkpoints/%s_optim_state.pt" % name)'''
    print("INFO: Checkpoint %s saved!" % name)

def delete_checkpoint(name):
    os.remove("./checkpoints/%s.tar" % name)
    '''os.remove("./checkpoints/%s_param_state.pt" % name)
    os.remove("./checkpoints/%s_optim_state.pt" % name)'''
    print("INFO: Checkpoint %s removed!" % name)

def load_checkpoint(model, optim, name):
    check = torch.load("./checkpoints/%s.tar" % name)
    model.load_state_dict(check['model_state_dict'])
    optim.load_state_dict(check['optimizer_state_dict'])
    print("INFO: Checkpoint %s loaded!" % name)
    return check["extra"]
    '''model.load_state_dict(torch.load("./checkpoints/%s_param_state.pt" % name))
    optim.load_state_dict(torch.load("./checkpoints/%s_optim_state.pt" % name))'''

screen_transforms = T.Compose([
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((84, 84), interpolation=Image.CUBIC),
    T.ToTensor()
])

def get_screen(env, device):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return screen_transforms(screen).unsqueeze(0).to(device)

def epsilon_greedy(state, policy_net, steps_done, n_actions, device, EPS_START, EPS_END, EPS_DECAY):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def plot_performance(durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    #plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model(policy_net, target_net, optimizer, memory, device, BATCH_SIZE, GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)