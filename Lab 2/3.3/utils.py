import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import cv2
import os
import imageio

# Configurazione device e seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Imposta seed per riproducibilità
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Wrapper per preprocessare le osservazioni
class CarRacingWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack=4, frame_skip=2, early_terminate_threshold=150):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=frame_stack)
        self.early_terminate_threshold = early_terminate_threshold
        
        # Contatori per early stopping più intelligente
        self.negative_reward_counter = 0
        self.total_reward = 0
        self.steps_since_positive = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.negative_reward_counter = 0
        self.total_reward = 0
        self.steps_since_positive = 0
        
        processed = self._preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return self._get_stacked_obs(), info

    def step(self, action):
        total_reward = 0
        done = False

        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Penalità più leggera per velocità bassa
            if -0.1 < reward < 0.1:
                reward -= 0.05  # Penalità più leggera
            
            total_reward += reward
            self.total_reward += reward
            done = terminated or truncated

            # Early stopping più intelligente
            if reward < 0:
                self.negative_reward_counter += 1
                self.steps_since_positive += 1
            else:
                self.negative_reward_counter = 0
                self.steps_since_positive = 0

            # Termina solo se veramente bloccato
            if self.early_terminate_threshold > 0:
                # Condizioni multiple per early stopping
                if (self.negative_reward_counter > self.early_terminate_threshold or
                    (self.steps_since_positive > 200 and self.total_reward < -50)):
                    done = True
                    # Penalità aggiuntiva per fallimento
                    total_reward -= 10
                    break

            if done:
                break

        processed = self._preprocess(obs)
        self.frames.append(processed)

        return self._get_stacked_obs(), total_reward, terminated, truncated or done, info
    
    def _preprocess(self, obs):
        # Conversione in grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize a 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # Normalizza tra 0 e 1
        return resized.astype(np.float32) / 255.0
    
    def _get_stacked_obs(self):
        # Stack dei frame come canali
        return np.stack(self.frames, axis=0)

# Classe per registrare video
class VideoRecorder:
    def __init__(self, env, output_path='videos'):
        self.env = env
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.frames = []
        
    def reset(self):
        self.frames = []
        
    def record_frame(self):
        frame = self.env.render()
        if frame is not None:
            self.frames.append(frame)
            
    def save_gif(self, filename, fps=30):
        if len(self.frames) > 0:
            path = os.path.join(self.output_path, filename)
            imageio.mimsave(path, self.frames, fps=fps)
            print(f"Video salvato: {path}")

#Per il plot
def smooth_rewards(rewards, window=100):
    if len(rewards) < window:
        return rewards 
    
    kernel = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode='valid')
    
    # Per allineare l'indice al centro della finestra
    padding = (len(rewards) - len(smoothed)) // 2
    smoothed = np.pad(smoothed, (padding, len(rewards) - len(smoothed) - padding), mode='constant', constant_values=np.nan)
    return smoothed