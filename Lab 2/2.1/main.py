import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium
import pygame
from model import PolicyNet
from reinforce import reinforce, run_episode

def main():
    # Per compatibilità 
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_

    # Inizializza PyGame
    _ = pygame.init()

    # Primo ambiente: rendering per visualizzare il comportamento iniziale
    env_render = gymnasium.make('CartPole-v1', render_mode='human')
    pygame.display.init()  # Per evitare crash di PyGame
    policy_initial = PolicyNet(env_render)
    for _ in range(10):
        run_episode(env_render, policy_initial)
    env_render.close()
    pygame.display.quit()

    # Ambiente per il training (senza rendering per velocizzare)
    seed = 2112
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gymnasium.make('CartPole-v1')
    env.reset(seed=seed)
    env_render = None  # Disattiviamo il rendering durante l'addestramento

    pygame.display.init()
    policy = PolicyNet(env)

    # Addestramento dell'agente con REINFORCE
    running_rewards = reinforce(
        policy, env, env_render,
        gamma=0.99,
        num_episodes=1000,
        eval_interval=50,     # Valutazione ogni 50 episodi
        eval_episodes=10,     # Su 10 episodi di valutazione
        temperature=1.0,      # Parametro di temperatura per esplorazione
        checkpoint_threshold=450  # Salva il modello se reward > 450
    )

    # Visualizzazione della progressione del training
    plt.figure(figsize=(10, 6))
    plt.plot(running_rewards)
    plt.xlabel("Episodi")
    plt.ylabel("Running Reward")
    plt.title("Progressione dell'Addestramento")
    plt.grid(True)
    plt.savefig("training_metrics_CartPole.png")  
    plt.show()

    env.close()
    pygame.display.quit()

    # Esecuzione finale dell'agente in ambiente con rendering
    env_render = gymnasium.make('CartPole-v1', render_mode='human')
    for _ in range(10):
        run_episode(env_render, policy, temperature=1.0)
    env_render.close()

if __name__ == "__main__":
    main()