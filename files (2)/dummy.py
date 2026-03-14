from ddqn import DDQNAgent
from pathlib import Path

ckpt_path = Path("models/gddqn.pt")
ckpt_path.parent.mkdir(exist_ok=True, parents=True)

agent = DDQNAgent(
    hidden_sizes=[128, 64],
    dropout=0.065,
    learning_rate=0.00015,
    gamma=0.978,
    epsilon_decay_steps=4106,
    batch_size=32,
    buffer_capacity=20000,
    target_update_freq=193,
    seed=42,
    checkpoint_path=str(ckpt_path)  # the agent uses this internally
)

# Use the agent's save method (creates a valid checkpoint)
agent.save()

print(f"Dummy G-DDQN checkpoint saved to {ckpt_path}")