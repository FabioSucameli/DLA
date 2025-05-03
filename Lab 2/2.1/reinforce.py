import numpy as np
import torch
from torch.distributions import Categorical
import wandb
    
#Data un'osservazione, restituisce l'azione scelta e il log-probabilità, 
#applicando un parametro di temperatura alla distribuzione della policy.
def select_action(env, obs, policy, temperature=1.0):
    probs = policy(obs)
    if temperature != 1.0:
        # Applichiamo una trasformazione per "scaldare" o "raffreddare" la distribuzione
        probs = torch.pow(probs, 1/temperature)
        probs /= probs.sum()  # Normalizziamo
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))

# Calcola i reward scontati per ogni passo dell'episodio.
def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    R = 0.0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R
        returns[i] = R
    return returns.copy()

#Esegue un episodio: resettando l'ambiente, raccoglie osservazioni, azioni, 
#log_prob e reward, fino al termine dell'episodio (o massimo di passi).   
def run_episode(env, policy, maxlen=500, temperature=1.0):   
    observations = []
    actions = []
    log_probs = []
    rewards = []
    
    (obs, info) = env.reset()
    for i in range(maxlen):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        (action, log_prob) = select_action(env, obs_tensor, policy, temperature)
        observations.append(obs_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)

# Addestra la policy usando l'algoritmo REINFORCE. Durante l'addestramento:
# Esegue periodicamente una valutazione dell'agente su M episodi.
# Registra le metriche con wandb.
# Salva il modello se raggiunge reward molto elevata.
def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=1000, 
              eval_interval=50, eval_episodes=10, temperature=1.0, checkpoint_threshold=450):
    
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    
    # Inizializza wandb
    wandb.init(project="cartpole", name="run_"+str(np.random.randint(1000)))
    
    running_rewards = [0.0]
    best_reward = -float('inf')
    
    policy.train()
    for episode in range(num_episodes):
        # Esegui un episodio e raccogli i dati
        observations, actions, log_probs, rewards = run_episode(env, policy, temperature=temperature)
        
        # Calcola i reward scontati
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        total_reward = sum(rewards)
        running_rewards.append(0.05 * total_reward + 0.95 * running_rewards[-1])
        
        # Standardizza i returns (attenua la varianza del gradiente)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calcola la loss e aggiorna la policy
        opt.zero_grad()
        loss = (- log_probs * returns).mean()
        loss.backward()
        opt.step()
        
        # Log con wandb
        wandb.log({
            "Loss": loss.item(),
            "Episode_Reward": total_reward,
            "Running_Reward": running_rewards[-1],
            "Episode_Length": len(rewards),
            "Episode": episode
        })
        
        # Valutazione periodica ogni 'eval_interval' episodi
        if episode % eval_interval == 0:
            policy.eval()
            total_rewards_eval = []
            episode_lengths_eval = []
            for _ in range(eval_episodes):
                _, _, _, rewards_eval = run_episode(env, policy, temperature=temperature)
                total_rewards_eval.append(sum(rewards_eval))
                episode_lengths_eval.append(len(rewards_eval))
            avg_reward = np.mean(total_rewards_eval)
            avg_length = np.mean(episode_lengths_eval)
            print(f"[Episodio {episode}] Valutazione: Reward media = {avg_reward:.2f}, Lunghezza media = {avg_length:.2f}")
            wandb.log({
                "Eval_Avg_Reward": avg_reward,
                "Eval_Avg_Length": avg_length,
                "Eval_Episode": episode
            })
            policy.train()
        
        # Salvataggio del modello
        if total_reward > checkpoint_threshold and total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy.state_dict(), f'best_policy_ep{episode}.pt')
            print(f"Checkpoint: Salvato modello all'episodio {episode} con reward {total_reward:.2f}")
        
        # Esempio di rendering ogni 100 episodi
        if env_render and episode % 100 == 0:
            policy.eval()
            run_episode(env_render, policy, temperature=temperature)
            policy.train()
    
    wandb.finish()
    policy.eval()
    return running_rewards