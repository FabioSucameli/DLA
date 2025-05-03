import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium
import pygame
from model import ValueNet, PolicyNet
from reinforce import reinforce, run_episode

def main():
    # Per compatibilità 
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_

    # Inizializza PyGame
    _ = pygame.init()

    # Primo ambiente: rendering per visualizzare il comportamento iniziale
    env_render = gymnasium.make('CartPole-v1', render_mode='human')
    pygame.display.init() # Per evitare crash di PyGame
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
    env_render = None # Disattiviamo il rendering durante l'addestramento

    # Esperimento 1: REINFORCE con standardizzazione semplice
    print("\n Esperimento 1: REINFORCE con standardizzazione semplice")
    policy_no_baseline = PolicyNet(env)
    rewards_no_baseline = reinforce(
        policy_no_baseline, env, env_render,
        gamma=0.99, num_episodes=1000,
        eval_interval=50, eval_episodes=10,
        temperature=1.0, checkpoint_threshold=450,
        use_baseline=False
    )

    # Esperimento 2: REINFORCE con rete di valore come baseline
    print("\n Esperimento 2: REINFORCE con rete di valore come baseline")
    policy_with_baseline = PolicyNet(env)
    value_net = ValueNet(env)
    rewards_with_baseline = reinforce(
        policy_with_baseline, env, env_render,
        gamma=0.99, num_episodes=1000,
        eval_interval=50, eval_episodes=10,
        temperature=1.0, checkpoint_threshold=450,
        use_baseline=True, value_net=value_net
    )

    # Visualizzazione della progressione del training
    plt.figure(figsize=(12, 8))
    plt.plot(rewards_no_baseline, label='Standardizzazione Semplice')
    plt.plot(rewards_with_baseline, label='Value Network Baseline')
    plt.xlabel("Episodi")
    plt.ylabel("Running Reward")
    plt.title("Confronto Progressione dell'Addestramento")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_baseline_methods.png")
    plt.show()

    # Esecuzione finale degli agenti
    env_render = gymnasium.make('CartPole-v1', render_mode='human')
    print("\nComportamento dell'agente con standardizzazione semplice:")
    for _ in range(3):
        run_episode(env_render, policy_no_baseline, temperature=1.0)

    print("\nComportamento dell'agente con value baseline:")
    for _ in range(3):
        run_episode(env_render, policy_with_baseline, temperature=1.0)

    # Chiusura degli ambienti
    env.close()
    env_render.close()
    pygame.display.quit()

if __name__ == "__main__":
    main()
