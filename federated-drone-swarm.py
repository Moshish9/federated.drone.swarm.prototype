"""
Federated Multi-Agent Drone Swarm (Prototype)
Author: Moshish Chaudhary


Goal: Simulate multiple drone agents learning from camera input (Vision Transformer)
and sharing knowledge via federated learning to create a shared global policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, ViTFeatureExtractor
from copy import deepcopy
import random
from PIL import Image

vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
for param in vit.parameters():
    param.requires_grad = False

class DronePolicy(nn.Module):
    def __init__(self, vision_dim=768, hidden_dim=256, action_dim=4):
        super(DronePolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class DroneAgent:
    def __init__(self, drone_id):
        self.id = drone_id
        self.policy = DronePolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.local_steps = 0

    def get_visual_features(self, image):
        inputs = feature_extractor(images=image, return_tensors='pt')
        with torch.no_grad():
            features = vit(**inputs).pooler_output
        return features

    def choose_action(self, features):
        probs = self.policy(features)
        action = torch.multinomial(probs, num_samples=1)
        return action.item()

    def local_update(self, image, reward):
        features = self.get_visual_features(image)
        action_probs = self.policy(features)
        loss = -torch.log(action_probs[0, random.randint(0, 3)]) * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.local_steps += 1

class FederatedServer:
    def __init__(self, global_policy):
        self.global_policy = global_policy

    def aggregate(self, local_agents):
        new_state = deepcopy(self.global_policy.state_dict())
        for key in new_state:
            local_tensors = [agent.policy.state_dict()[key] for agent in local_agents]
            new_state[key] = sum(local_tensors) / len(local_tensors)
        self.global_policy.load_state_dict(new_state)
        print("[Server] Federated aggregation complete.")

    def distribute(self, local_agents):
        for agent in local_agents:
            agent.policy.load_state_dict(self.global_policy.state_dict())

if __name__ == "__main__":
    NUM_DRONES = 3
    NUM_ROUNDS = 3
    DRONE_AGENTS = [DroneAgent(drone_id=i) for i in range(NUM_DRONES)]
    global_policy = DronePolicy()
    server = FederatedServer(global_policy)

    fake_images = [Image.new('RGB', (224, 224), color=(i*40, i*40, i*40)) for i in range(NUM_DRONES)]

    for round in range(NUM_ROUNDS):
        print(f"\n=== Federated Round {round+1} ===")
        for agent, img in zip(DRONE_AGENTS, fake_images):
            reward = random.uniform(0, 1)
            agent.local_update(img, reward)
            print(f"Drone {agent.id} updated locally. Total steps: {agent.local_steps}")
        server.aggregate(DRONE_AGENTS)
        server.distribute(DRONE_AGENTS)

    print("\n✅ Simulation complete. Drones now share a global policy.")
