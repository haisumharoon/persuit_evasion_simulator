import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

torch.set_num_threads(4)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PursuitEvasionEnv:
    def __init__(self, n_pursuers=4, arena=100.0, dt=1.0,
                 v_pursuer=1.0, v_evader=2.0,
                 omega_max_p=1.0, omega_max_e=2.0,
                 capture_dist=5.0, ep_len=256):
        self.N = n_pursuers
        self.arena = arena
        self.dt = dt
        self.v_p = v_pursuer
        self.v_e = v_evader
        self.omega_max_p = omega_max_p
        self.omega_max_e = omega_max_e
        self.capture_dist = capture_dist
        self.ep_len = ep_len
        self.obs_dim = 11
        self.act_dim = 1

    def reset(self):
        self.t = 0
        self.px = np.random.uniform(0, self.arena, self.N)
        self.py = np.random.uniform(0, self.arena, self.N)
        self.pphi = np.random.uniform(0, 2*np.pi, self.N)
        self.pomega = np.zeros(self.N)
        self.ex = np.random.uniform(0, self.arena)
        self.ey = np.random.uniform(0, self.arena)
        self.ephi = np.random.uniform(0, 2*np.pi)
        self.eomega = 0.0
        self.captured = False
        return self._get_obs()

    def _dist(self, x1, y1, x2, y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def _get_obs(self):
        obs = []
        for i in range(self.N):
            others = [j for j in range(self.N) if j != i]
            d_others = [self._dist(self.px[i], self.py[i], self.px[j], self.py[j]) for j in others]
            o = np.array([
                self.px[i], self.py[i], self.v_p, self.pomega[i],
                self.ex, self.ey, self.v_e, self.eomega,
                *d_others
            ], dtype=np.float32)
            obs.append(o)
        return obs

    def _evader_action(self):
        dists = [self._dist(self.ex, self.ey, self.px[i], self.py[i]) for i in range(self.N)]
        nearest = np.argmin(dists)
        away_angle = np.arctan2(self.ey - self.py[nearest], self.ex - self.px[nearest])
        angle_diff = (away_angle - self.ephi + np.pi) % (2*np.pi) - np.pi
        omega = np.clip(angle_diff, -self.omega_max_e, self.omega_max_e)
        return omega

    def step(self, actions):
        for i in range(self.N):
            omega = np.clip(actions[i], -self.omega_max_p, self.omega_max_p)
            self.pomega[i] = omega
            self.px[i] += self.v_p * np.cos(self.pphi[i]) * self.dt
            self.py[i] += self.v_p * np.sin(self.pphi[i]) * self.dt
            self.pphi[i] = (self.pphi[i] + omega * self.dt) % (2*np.pi)
            self.px[i] = np.clip(self.px[i], 0, self.arena)
            self.py[i] = np.clip(self.py[i], 0, self.arena)

        e_omega = self._evader_action()
        self.eomega = e_omega
        self.ex += self.v_e * np.cos(self.ephi) * self.dt
        self.ey += self.v_e * np.sin(self.ephi) * self.dt
        self.ephi = (self.ephi + e_omega * self.dt) % (2*np.pi)
        self.ex = np.clip(self.ex, 0, self.arena)
        self.ey = np.clip(self.ey, 0, self.arena)

        d_min = min(self._dist(self.px[i], self.py[i], self.ex, self.ey) for i in range(self.N))
        d_o = np.sqrt(2) * self.arena
        reward = -(1.0/d_o) * min(d_min, d_o)
        self.captured = d_min < self.capture_dist
        self.t += 1
        done = self.captured or (self.t >= self.ep_len)

        return self._get_obs(), reward, done, self.captured

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh()
        )
    def forward(self, obs):
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, joint_obs_dim, joint_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim + joint_act_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, joint_obs, joint_act):
        x = torch.cat([joint_obs, joint_act], dim=-1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=capacity)
    def push(self, obs, act, rew, next_obs, done):
        self.buffer.append((obs, act, rew, next_obs, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return obs, act, rew, next_obs, done
    def __len__(self):
        return len(self.buffer)

class MADDPG:
    def __init__(self, n_agents, obs_dim, act_dim=1, gamma=0.99, tau=0.1, lr=1e-4):
        self.n = n_agents
        self.gamma = gamma
        self.tau = tau
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        joint_obs_dim = obs_dim * n_agents
        joint_act_dim = act_dim * n_agents

        self.actors = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.actors_target = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.critics = [Critic(joint_obs_dim, joint_act_dim).to(device) for _ in range(n_agents)]
        self.critics_target = [Critic(joint_obs_dim, joint_act_dim).to(device) for _ in range(n_agents)]

        for i in range(n_agents):
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

        self.actor_opt = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.critic_opt = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]

        self.buffer = ReplayBuffer()

    def act(self, obs_list, noise_scale=0.1):
        actions = []
        for i in range(self.n):
            o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                a = self.actors[i](o).cpu().numpy()[0]
            a = a + noise_scale * np.random.randn(self.act_dim)
            a = np.clip(a, -1, 1)
            actions.append(a)
        return actions

    def update(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return

        obs, act, rew, next_obs, done = self.buffer.sample(batch_size)
        obs = np.array(obs)
        act = np.array(act)
        next_obs = np.array(next_obs)
        rew = np.array(rew)
        done = np.array(done, dtype=np.float32)

        obs_t = torch.FloatTensor(obs).to(device)
        act_t = torch.FloatTensor(act).to(device)
        next_obs_t = torch.FloatTensor(next_obs).to(device)
        rew_t = torch.FloatTensor(rew).to(device)
        done_t = torch.FloatTensor(done).to(device)

        joint_obs = obs_t.reshape(batch_size, -1)
        joint_act = act_t.reshape(batch_size, -1)
        joint_next_obs = next_obs_t.reshape(batch_size, -1)

        next_actions = []
        for i in range(self.n):
            next_actions.append(self.actors_target[i](next_obs_t[:, i, :]))
        joint_next_act = torch.cat(next_actions, dim=-1)

        for i in range(self.n):
            with torch.no_grad():
                q_next = self.critics_target[i](joint_next_obs, joint_next_act).squeeze(-1)
                y = rew_t + self.gamma * (1 - done_t) * q_next
            q_val = self.critics[i](joint_obs, joint_act).squeeze(-1)
            critic_loss = nn.functional.mse_loss(q_val, y)

            self.critic_opt[i].zero_grad()
            critic_loss.backward()
            self.critic_opt[i].step()

            act_pred = [act_t[:, j, :] if j != i else self.actors[i](obs_t[:, i, :]) for j in range(self.n)]
            joint_act_pred = torch.cat(act_pred, dim=-1)
            actor_loss = -self.critics[i](joint_obs, joint_act_pred).mean()

            self.actor_opt[i].zero_grad()
            actor_loss.backward()
            self.actor_opt[i].step()

        for i in range(self.n):
            for p, tp in zip(self.actors[i].parameters(), self.actors_target[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critics[i].parameters(), self.critics_target[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

def train(n_episodes=700, episode_len=256, batch_size=256, log_every=5, update_every=4):
    env = PursuitEvasionEnv(n_pursuers=4, ep_len=episode_len)
    maddpg = MADDPG(n_agents=4, obs_dim=env.obs_dim, act_dim=1)

    capture_history = []
    capture_rate_curve = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        noise_scale = max(0.05, 0.3 * (1 - ep / n_episodes))

        for t in range(episode_len):
            actions = maddpg.act(obs, noise_scale=noise_scale)
            next_obs, reward, done, captured = env.step([a[0] for a in actions])

            maddpg.buffer.push(obs, actions, reward, next_obs, done)
            obs = next_obs

            if t % update_every == 0:
                maddpg.update(batch_size=batch_size)

            if done:
                break

        capture_history.append(1 if captured else 0)
        window = capture_history[-50:]
        rate = 100 * sum(window) / len(window)
        capture_rate_curve.append(rate)

        if ep % log_every == 0:
            print(f"Episode {ep}/{n_episodes} | rolling capture rate: {rate:.1f}%", flush=True)

    return capture_rate_curve, maddpg, env

def record_rollout(maddpg, env, max_steps=256):
    """Run one greedy (no-exploration) episode and record positions for animation."""
    obs = env.reset()
    px_hist, py_hist, ex_hist, ey_hist = [], [], [], []
    for t in range(max_steps):
        px_hist.append(env.px.copy())
        py_hist.append(env.py.copy())
        ex_hist.append(env.ex)
        ey_hist.append(env.ey)
        actions = maddpg.act(obs, noise_scale=0.0)  
        obs, reward, done, captured = env.step([a[0] for a in actions])
        if done:
            px_hist.append(env.px.copy())
            py_hist.append(env.py.copy())
            ex_hist.append(env.ex)
            ey_hist.append(env.ey)
            break
    return px_hist, py_hist, ex_hist, ey_hist, captured

def animate_rollout(px_hist, py_hist, ex_hist, ey_hist, arena=100.0, filename="pursuit_animation.gif"):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, arena)
    ax.set_ylim(0, arena)
    ax.set_title("Pursuit-Evasion Rollout")

    pursuer_dots, = ax.plot([], [], 'bo', markersize=10, label='Pursuers')
    evader_dot, = ax.plot([], [], 'ro', markersize=10, label='Evader')
    ax.legend(loc='upper right')

    def update_frame(frame):
        pursuer_dots.set_data(px_hist[frame], py_hist[frame])
        evader_dot.set_data([ex_hist[frame]], [ey_hist[frame]])
        return pursuer_dots, evader_dot

    ani = animation.FuncAnimation(fig, update_frame, frames=len(px_hist), interval=80, blit=True)
    ani.save(filename, writer='pillow')
    print(f"Saved animation to {filename}")
    plt.close(fig)

if __name__ == "__main__":
    curve, trained_maddpg, env = train(n_episodes=700, episode_len=256, batch_size=256)

    plt.figure()
    plt.plot(curve)
    plt.xlabel("episodes")
    plt.ylabel("capture rate (%)")
    plt.title("MADDPG Pursuit-Evasion: Capture Rate Convergence")
    plt.savefig("capture_rate_curve.png")
    plt.close()
    print("Saved capture_rate_curve.png")

    px_hist, py_hist, ex_hist, ey_hist, captured = record_rollout(trained_maddpg, env)
    print(f"Rollout captured: {captured}, steps: {len(px_hist)}")
    animate_rollout(px_hist, py_hist, ex_hist, ey_hist, arena=env.arena)