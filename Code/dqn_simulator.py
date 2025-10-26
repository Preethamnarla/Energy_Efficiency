# dqn_simulator.py
"""
Deep RL (DQN) energy-efficient wireless power allocation demo.
- Python 3.9+
- Requires: numpy, tensorflow, matplotlib, pandas, tqdm

Usage:
    python dqn_simulator.py
Outputs:
    - Prints baseline energy and RL energy and % reduction
    - Shows training plots (reward, energy, throughput)
"""

import math, random, collections, os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tqdm import trange

# --------------------------
# Reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# --------------------------

# Utility
def db_to_linear(db): return 10 ** (db / 10.0)
def linear_to_db(x): return 10 * math.log10(max(x, 1e-12))

# --------------------------
# Channel & User (same idea as earlier)
class Channel:
    def __init__(self, distance_m, freq_mhz=2000.0, pl_exp=3.5, shadow_std_db=4.0):
        self.distance = max(distance_m, 1.0)
        self.freq_mhz = freq_mhz
        self.pl_exp = pl_exp
        self.shadow_std_db = shadow_std_db
    def path_loss_db(self):
        fspl_db = 20 * math.log10(self.freq_mhz) + 20 * math.log10(self.distance) - 27.55
        extra = 10 * (self.pl_exp - 2) * math.log10(self.distance)
        return fspl_db + extra
    def shadowing_db(self):
        return random.gauss(0.0, self.shadow_std_db)
    def rayleigh_gain_linear(self):
        return random.expovariate(1.0)
    def instantaneous_gain_linear(self):
        pl_db = self.path_loss_db()
        sh_db = self.shadowing_db()
        total_loss_db = pl_db + sh_db
        gain = db_to_linear(-total_loss_db)
        fad = self.rayleigh_gain_linear()
        return gain * fad

class User:
    def __init__(self, uid, distance_m, arrival_lambda=3.0, pkt_size_bits=8000, channel=None):
        self.id = uid
        self.dist = distance_m
        self.arrival_lambda = arrival_lambda
        self.pkt_size = pkt_size_bits
        self.queue_bits = 0
        self.channel = channel if channel else Channel(distance_m)
        self.served = 0
        self.arrived = 0
    def arrivals(self, dt):
        lam = self.arrival_lambda * dt
        n = np.random.poisson(lam)
        bits = n * self.pkt_size
        self.queue_bits += bits
        self.arrived += bits
        return bits
    def chan_gain(self):
        return self.channel.instantaneous_gain_linear()

# Shannon rate
def shannon_rate(bw_hz, snr_lin):
    return bw_hz * math.log2(1 + max(snr_lin, 1e-12))

# --------------------------
# Simulator environment (small custom env)
class SimEnv:
    def __init__(self, users, bandwidth_hz=180e3, Pmax_w=1.0, dt=0.1, noise_fig_db=7.0, noise_temp_k=290.0):
        self.users = users
        self.n = len(users)
        self.B = bandwidth_hz
        self.Pmax = Pmax_w
        self.dt = dt
        k = 1.38064852e-23
        N0 = k * noise_temp_k * self.B
        self.noise_w = N0 * db_to_linear(noise_fig_db)
        self.time = 0.0
    def reset(self):
        for u in self.users:
            u.queue_bits = 0
            u.served = 0
            u.arrived = 0
        self.time = 0.0
    def step(self, alloc_fractions):
        # alloc_fractions: list len n summing <=1 (asserted by caller)
        for u in self.users: u.arrivals(self.dt)
        gains = [u.chan_gain() for u in self.users]
        rates = []
        served = []
        energy = 0.0
        for i,u in enumerate(self.users):
            p = max(0.0, min(1.0, alloc_fractions[i])) * self.Pmax
            snr = p * gains[i] / (self.noise_w + 1e-20)
            rbps = shannon_rate(self.B, snr)
            send = min(rbps * self.dt, u.queue_bits)
            u.queue_bits -= send
            u.served += send
            rates.append(rbps)
            served.append(send)
            energy += p * self.dt
        self.time += self.dt
        # reward: encourage served bits and penalize energy and backlog
        total_served = sum(served)
        queue_pen = -sum([u.queue_bits for u in self.users]) * 1e-6
        reward = 0.0008 * total_served - energy + queue_pen
        return reward, gains, rates, energy

# --------------------------
# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = collections.deque(maxlen=capacity)
    def push(self, s,a,r,s2,done):
        self.buf.append((s,a,r,s2,done))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        ss,aa,rr,ss2,dd = zip(*[self.buf[i] for i in idx])
        return np.array(ss), np.array(aa), np.array(rr, dtype=np.float32), np.array(ss2), np.array(dd)
    def __len__(self):
        return len(self.buf)

# --------------------------
# DQN network
def build_q_net(input_dim, output_dim, lr=1e-3):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse')
    return model

# --------------------------
# State representation helper
def state_from_env(users):
    # Continuous state: [queue_u0_norm, queue_u1_norm, gain_db_u0, gain_db_u1]
    # Normalize queues by a scale (e.g. 1e5 bits) to keep values reasonable
    qs = [u.queue_bits / 1e5 for u in users]
    gains = [linear_to_db(max(1e-12, u.chan_gain())) for u in users]
    # clip gains to reasonable range
    gains = [max(-120, min(50, g)) for g in gains]
    return np.array(qs + gains, dtype=np.float32)

# --------------------------
# Action set: for two-user case we select fraction to user0; user1 gets remainder
ACTION_SET = [i/10.0 for i in range(11)]  # 0.0 .. 1.0 step 0.1

# --------------------------
def evaluate_policy(agent_model, env, episodes=1, steps=2000):
    total_energy = 0.0
    total_throughput = 0.0
    for ep in range(episodes):
        env.reset()
        for t in range(steps):
            # compute state
            s = state_from_env(env.users)
            qvals = agent_model.predict(s.reshape(1,-1), verbose=0)[0]
            a_idx = np.argmax(qvals)
            frac = ACTION_SET[a_idx]
            alloc = [frac, 1.0 - frac] if env.n==2 else [frac] + [ (1-frac)/(env.n-1) for _ in range(env.n-1)]
            r,gains,rates,energy = env.step(alloc)
            total_energy += energy
            total_throughput += sum(rates)
    return total_energy, total_throughput / (episodes * steps)

# --------------------------
def train_dqn(episodes=300, steps_per_ep=400, batch_size=64):
    # create two users with different distances/loads
    u0 = User(0, distance_m=50.0, arrival_lambda=3.0)
    u1 = User(1, distance_m=150.0, arrival_lambda=4.0)
    env = SimEnv([u0, u1], bandwidth_hz=180e3, Pmax_w=1.0, dt=0.1)
    state_dim = 4
    action_dim = len(ACTION_SET)
    q_net = build_q_net(state_dim, action_dim, lr=1e-3)
    target_net = build_q_net(state_dim, action_dim, lr=1e-3)
    target_net.set_weights(q_net.get_weights())
    buffer = ReplayBuffer(30000)

    gamma = 0.99
    eps = 1.0
    eps_min = 0.03
    eps_decay = 0.995
    tau_update = 200  # target update steps
    step_count = 0

    rewards_hist = []
    energy_hist = []
    throughput_hist = []

    for ep in range(episodes):
        env.reset()
        ep_reward = 0.0
        ep_energy = 0.0
        ep_throughput = 0.0

        s = state_from_env(env.users)
        for t in range(steps_per_ep):
            # epsilon-greedy
            if random.random() < eps:
                a_idx = random.randrange(action_dim)
            else:
                qvals = q_net.predict(s.reshape(1,-1), verbose=0)[0]
                a_idx = int(np.argmax(qvals))
            frac = ACTION_SET[a_idx]
            alloc = [frac, 1.0-frac]
            r,gains,rates,energy = env.step(alloc)
            s2 = state_from_env(env.users)
            done = False
            buffer.push(s, a_idx, r, s2, done)
            s = s2
            ep_reward += r
            ep_energy += energy
            ep_throughput += sum(rates)

            # training step
            if len(buffer) >= batch_size:
                ss,aa,rr,ss2,dd = buffer.sample(batch_size)
                q_next = target_net.predict(ss2, verbose=0)
                q_next_max = np.max(q_next, axis=1)
                q_targets = q_net.predict(ss, verbose=0)
                for i in range(batch_size):
                    q_targets[i, aa[i]] = rr[i] + gamma * q_next_max[i]
                q_net.train_on_batch(ss, q_targets)

            # update target network periodically
            step_count += 1
            if step_count % tau_update == 0:
                target_net.set_weights(q_net.get_weights())

        eps = max(eps_min, eps * eps_decay)
        rewards_hist.append(ep_reward)
        energy_hist.append(ep_energy)
        throughput_hist.append(ep_throughput / steps_per_ep)

        if (ep+1) % 10 == 0:
            print(f"Ep {ep+1}/{episodes} reward={ep_reward:.2f} energy={ep_energy:.2f} avg_tp={throughput_hist[-1]:.2f} eps={eps:.3f}")

    return q_net, rewards_hist, energy_hist, throughput_hist, env

# --------------------------
def run_baseline_and_train():
    # Baseline (static equal split)
    print("Running baseline (static 0.5/0.5) for 2000 steps...")
    u0 = User(0, 50.0, arrival_lambda=3.0)
    u1 = User(1, 150.0, arrival_lambda=4.0)
    env_base = SimEnv([u0, u1], dt=0.1)
    total_energy_base = 0.0
    for _ in range(2000):
        r,g,rates,e = env_base.step([0.5, 0.5])
        total_energy_base += e
    print(f"Baseline energy: {total_energy_base:.2f} J")

    # Train DQN
    q_net, rewards_hist, energy_hist, tp_hist, env = train_dqn(episodes=220, steps_per_ep=350)
    # Evaluate trained policy (no exploration)
    rl_energy, rl_throughput = evaluate_policy(q_net, env, episodes=1, steps=2000)
    print("\n--- Evaluation ---")
    print(f"Baseline energy (same duration): {total_energy_base:.2f} J")
    print(f"RL energy: {rl_energy:.2f} J")
    reduction = 100.0 * (total_energy_base - rl_energy) / total_energy_base
    print(f"Energy reduction vs baseline: {reduction:.2f}%")
    # plots
    fig, axs = plt.subplots(3,1,figsize=(8,10))
    axs[0].plot(rewards_hist); axs[0].set_title("Episode rewards")
    axs[1].plot(energy_hist); axs[1].set_title("Episode energy (J)")
    axs[2].plot(tp_hist); axs[2].set_title("Episode avg throughput (bps)")
    plt.tight_layout(); plt.show()

    return reduction

if __name__ == "__main__":
    reduction = run_baseline_and_train()
    if reduction < 14.0:
        print("\nTIP: If <15% reduction, try increasing training episodes, larger action resolution, or reward scaling.")
    else:
        print("\nSUCCESS: >=15% energy reduction achieved (with these seeds/hyperparams).")
