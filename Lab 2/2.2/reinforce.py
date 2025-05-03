import numpy as np
import torch
from torch.distributions import Categorical
import wandb


# Data un'osservazione, restituisce l'azione scelta e il log-probabilità,
# applicando un parametro di temperatura alla distribuzione della policy.
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


# Esegue un episodio: resettando l'ambiente, raccoglie osservazioni, azioni,
# log_prob e reward, fino al termine dell'episodio (o massimo di passi).
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


# Addestra la policy usando l'algoritmo REINFORCE, con possibilità di usare una rete di valore come baseline.
# Durante l'addestramento:
# Se use_baseline=True, allena una rete di valore separata per stimare V(s) e usarla come baseline.
# In alternativa, applica una standardizzazione dei ritorni all'interno dell'episodio.
# Registra le metriche con wandb.
# Salva i modelli se raggiunge reward molto elevata.
def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=1000, 
              eval_interval=50, eval_episodes=10, temperature=1.0, 
              checkpoint_threshold=450, use_baseline=False, value_net=None):
    
    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=1e-2) if use_baseline else None

    # Inizializza wandb
    wandb.init(project="cartpole", name="run_"+str(np.random.randint(1000)))
    
    running_rewards = [0.0]
    best_reward = -float('inf')
    
    policy.train()
    if use_baseline:
        value_net.train()
    
    for episode in range(num_episodes):
        observations, actions, log_probs, rewards = run_episode(env, policy, temperature=temperature)

        # Calcola i reward scontati
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        total_reward = sum(rewards)
        running_rewards.append(0.05 * total_reward + 0.95 * running_rewards[-1])

        obs_tensor = torch.stack(observations)

        if use_baseline:
            # Calcola i valori stimati e aggiorna la rete di valore
            values = value_net(obs_tensor).squeeze()
            value_opt.zero_grad()
            value_loss = torch.nn.functional.mse_loss(values, returns)
            value_loss.backward()
            value_opt.step()
            # Calcola l'advantage
            advantages = returns - values.detach()
        else:
            # Solo standardizzazione dei ritorni
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            value_loss = torch.tensor(0.0)  # placeholder per logging

        # Aggiornamento della policy
        policy_opt.zero_grad()
        policy_loss = (-log_probs * advantages).mean()
        policy_loss.backward()
        policy_opt.step()

        # Log su wandb
        wandb.log({
            "Policy_Loss": policy_loss.item(),
            "Value_Loss": value_loss.item() if use_baseline else 0.0,
            "Episode_Reward": total_reward,
            "Running_Reward": running_rewards[-1],
            "Episode_Length": len(rewards),
            "Episode": episode,
            "Using_Baseline": use_baseline
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
            if use_baseline:
                torch.save(value_net.state_dict(), f'best_value_net_ep{episode}.pt')
            print(f"Checkpoint: Salvato modello all'episodio {episode} con reward {total_reward:.2f}")
        
        # Esempio di rendering ogni 100 episodi
        if env_render and episode % 100 == 0:
            policy.eval()
            run_episode(env_render, policy, temperature=temperature)
            policy.train()

    wandb.finish()
    policy.eval()
    if use_baseline:
        value_net.eval()
    
    return running_rewards
