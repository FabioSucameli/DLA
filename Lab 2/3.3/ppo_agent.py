import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import device  

# Rete Actor-Critic con CNN
class ActorCriticCNN(nn.Module):
    def __init__(self, num_actions, frame_stack=4):
        super().__init__()
        
        # Shared CNN backbone
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calcola la dimensione dopo le convoluzioni
        conv_out_size = self._get_conv_out_size((frame_stack, 84, 84))
        
        # Shared FC layer
        self.fc = nn.Linear(conv_out_size, 512)
        
        # Actor head (policy)
        self.actor = nn.Linear(512, num_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(512, 1)
        
    def _get_conv_out_size(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))
    
    def forward(self, x):
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # Shared FC
        x = F.relu(self.fc(x))
        
        # Actor and critic outputs
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return action_probs, value

# Buffer ottimizzato per memorizzare le esperienze
class RolloutBuffer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        # Converti stato in tensore subito per efficienza
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self, gamma, gae_lambda, last_value=0):
        # Calcola advantages usando GAE
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                # Usa il valore predetto se non è terminale
                next_value = last_value if not self.dones[t] else 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # Calcola returns
        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(device)
        
        # Normalizza advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Converti tutto in tensori (più efficiente se già torch tensors)
        if isinstance(self.states[0], torch.Tensor):
            states = torch.stack(self.states).to(device)
        else:
            states = torch.tensor(np.array(self.states), dtype=torch.float32).to(device)
            
        actions = torch.tensor(self.actions, dtype=torch.long).to(device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device)
        
        return states, actions, log_probs, returns, advantages

# Learning rate scheduler
class LinearSchedule:
    def __init__(self, initial_lr, final_lr, num_updates):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.num_updates = num_updates
        self.current_update = 0
        
    def get_lr(self):
        frac = self.current_update / self.num_updates
        return self.initial_lr + frac * (self.final_lr - self.initial_lr)
    
    def step(self):
        self.current_update += 1

# Agente PPO migliorato
class PPOAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, value_clip_epsilon=0.2, epochs=10, 
                 batch_size=64, num_updates=1000):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_clip_epsilon = value_clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Modello
        self.model = ActorCriticCNN(env.action_space.n).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.lr_scheduler = LinearSchedule(lr, lr * 0.1, num_updates)
        
        # Buffer
        self.buffer = RolloutBuffer()
    
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, value = self.model(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self):
        # Ottieni il valore dell'ultimo stato per la baseline
        if len(self.buffer.states) > 0 and not self.buffer.dones[-1]:
            with torch.no_grad():
                last_state = self.buffer.states[-1]
                if not isinstance(last_state, torch.Tensor):
                    last_state = torch.tensor(last_state, dtype=torch.float32)
                last_state = last_state.unsqueeze(0).to(device)
                _, last_value = self.model(last_state)
                last_value = last_value.item()
        else:
            last_value = 0
            
        # Ottieni dati dal buffer
        states, actions, old_log_probs, returns, advantages = self.buffer.get(
            self.gamma, self.gae_lambda, last_value
        )
        
        # Salva i vecchi valori per il clipping
        with torch.no_grad():
            _, old_values = self.model(states)
            old_values = old_values.squeeze()
        
        # Mini-batch training
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass
                action_probs, values = self.model(batch_states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                
                # Calcola ratio per PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss con clipping (come nel paper PPO)
                values_clipped = batch_old_values + torch.clamp(
                    values.squeeze() - batch_old_values,
                    -self.value_clip_epsilon,
                    self.value_clip_epsilon
                )
                value_loss_1 = F.mse_loss(values.squeeze(), batch_returns)
                value_loss_2 = F.mse_loss(values_clipped, batch_returns)
                value_loss = torch.max(value_loss_1, value_loss_2)
                
                # Entropy bonus per esplorazione
                entropy = dist.entropy().mean()
                
                # Loss totale
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        
        # Aggiorna learning rate
        new_lr = self.lr_scheduler.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.lr_scheduler.step()
        
        # Reset buffer
        self.buffer.reset()