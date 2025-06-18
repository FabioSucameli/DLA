import numpy as np
import torch
import gymnasium as gym
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
from utils import CarRacingWrapper, VideoRecorder, smooth_rewards, set_seed, device
from ppo_agent import PPOAgent

def train_ppo(num_episodes=1000, max_steps_per_episode=1000, 
              rollout_steps=2048, eval_interval=50, eval_episodes=5):
    
    # Inizializza wandb
    run_name = f"ppo_carracing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="carracing-ppo", name=run_name, config={
        "algorithm": "PPO",
        "num_episodes": num_episodes,
        "max_steps": max_steps_per_episode,
        "rollout_steps": rollout_steps,
        "device": str(device),
        "seed": 42
    })
    
    # Ambiente con wrapper
    env = gym.make('CarRacing-v3', continuous=False)
    env = CarRacingWrapper(env, early_terminate_threshold=150)
    
    # Agente con learning rate più conservativo
    agent = PPOAgent(env, lr=2e-4, num_updates=num_episodes // 3) 

    # Training loop
    episode = 0
    total_steps = 0
    best_reward = -float('inf')
    episode_rewards = []
    
    while episode < num_episodes:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Raccogli esperienza con early stopping
        rollout_count = 0
        while rollout_count < rollout_steps and not done:

            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Aggiungi al buffer
            agent.buffer.add(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            rollout_count += 1
            
            # Early stopping se l'episodio è terminato
            if done or episode_steps >= max_steps_per_episode:
                episode += 1
                episode_rewards.append(episode_reward)
                
                # Log metriche
                current_lr = agent.lr_scheduler.get_lr()
                wandb.log({
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "episode_steps": episode_steps,
                    "total_steps": total_steps,
                    "avg_reward_100": np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
                    "learning_rate": current_lr
                })
                
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {episode_steps}, LR: {current_lr:.6f}")
                
                # Reset per nuovo episodio se non abbiamo finito il training
                if episode < num_episodes and rollout_count < rollout_steps:
                    state, _ = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    done = False
        
        # Aggiorna policy
        agent.update()
        
        # Valutazione con video
        if episode % eval_interval == 0 and episode > 0:
            eval_rewards = []
            
            # Crea ambiente per valutazione con render
            eval_env = gym.make('CarRacing-v3', continuous=False, render_mode='rgb_array')
            eval_env = CarRacingWrapper(eval_env)
            video_recorder = VideoRecorder(eval_env)
            
            for eval_ep in range(eval_episodes):
                state, _ = eval_env.reset()
                eval_reward = 0
                done = False
                steps = 0
                
                # Registra solo il primo episodio di valutazione
                if eval_ep == 0:
                    video_recorder.reset()
                
                while not done and steps < max_steps_per_episode:
                    action, _, _ = agent.select_action(state)
                    state, reward, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    eval_reward += reward
                    steps += 1
                    
                    # Registra frame per il video
                    if eval_ep == 0:
                        video_recorder.record_frame()
                
                eval_rewards.append(eval_reward)
            
            # Salva video del primo episodio
            video_recorder.save_gif(f'eval_episode_{episode}.gif', fps=15)
            eval_env.close()
            
            avg_eval_reward = np.mean(eval_rewards)
            wandb.log({
                "eval_avg_reward": avg_eval_reward,
                "eval_episode": episode
            })
            
            print(f"\n[Evaluation] Episode {episode}, Avg Reward: {avg_eval_reward:.2f}\n")
            
            # Salva il modello se è il migliore
            if avg_eval_reward > best_reward:
                best_reward = avg_eval_reward
                torch.save({
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'episode': episode,
                    'best_reward': best_reward
                }, f'best_ppo_carracing_{episode}.pt')
                print(f"Nuovo miglior modello salvato! Reward: {avg_eval_reward:.2f}")
    
    # Chiudi ambiente e wandb
    env.close()
    wandb.finish()
    
    # Plot dei risultati
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    window = 100
    if len(episode_rewards) >= window:
        smoothed = smooth_rewards(episode_rewards, window=100)
        plt.plot(smoothed)
        plt.title(f'Smoothed Rewards (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('ppo_carracing_training.png')
    plt.show()
    
    return agent

# Funzione per testare l'agente addestrato
def test_agent(agent, num_episodes=5, render=True, save_video=True):
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make('CarRacing-v3', continuous=False, render_mode=render_mode)
    env = CarRacingWrapper(env)
    
    if save_video and not render:
        video_recorder = VideoRecorder(env)
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        if save_video and not render and episode == 0:
            video_recorder.reset()
        
        while not done and steps < 1000:
            action, _, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            if save_video and not render and episode == 0:
                video_recorder.record_frame()
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}, Steps: {steps}")
    
    if save_video and not render:
        video_recorder.save_gif('test_run.gif', fps=15)
    
    env.close()
    
    print(f"\nTest Summary:")
    print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Max Reward: {np.max(test_rewards):.2f}")
    print(f"Min Reward: {np.min(test_rewards):.2f}")

if __name__ == "__main__":
    # Training
    print("Starting PPO training on CarRacing...")
    agent = train_ppo(
        num_episodes=1000,
        max_steps_per_episode=1000,
        rollout_steps=2048,
        eval_interval=20,
        eval_episodes=3
    )
    
    # Test dell'agente addestrato
    print("\nTesting trained agent...")
    test_agent(agent, num_episodes=5, render=True, save_video=False)
