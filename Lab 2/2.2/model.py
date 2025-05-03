import torch.nn as nn
import torch.nn.functional as F


# Definizione della rete neurale per la policy
class PolicyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)
    
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s

# Definizione della rete neurale per la funzione valore
class ValueNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = self.fc2(s)  # Output scalare
        return s