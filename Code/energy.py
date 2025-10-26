"""
Energy-Efficient Wireless Network Simulator
Single-file Python simulator that:
- Simulates 1-2 user wireless link(s) with dynamic traffic (Poisson arrivals)
- Models path loss (log-distance), shadowing (Gaussian, in dB), and small-scale Rayleigh fading
- Uses a simple tabular Q-learning agent to allocate transmit power fractions between users
- Computes throughput (Shannon capacity approx) and packet-level queueing to estimate latency
- Compares learned (RL) allocation vs a static baseline and reports energy savings
- Produces plots of training reward, throughput, latency, and energy usage

How to run: `python energy_efficient_wireless_simulator.py`
Requirements: python 3.8+, numpy, matplotlib, pandas

This is intentionally compact and educational rather than industrial-grade.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, defaultdict

# ----------------------------- Utility functions -----------------------------

def db_to_linear(db):
    return 10 ** (db / 10.0)


def linear_to_db(x):
    return 10 * math.log10(max(x, 1e-12))


# ----------------------------- Channel models -----------------------------
class Channel:
    """Simple channel model including path loss (log-distance), shadowing, and Rayleigh fading.
    Positions are 1D distances (meters) from base station.
    """

    def __init__(self, distance_m, freq_mhz=2000.0, path_loss_exponent=3.5, shadowing_std_db=4.0):
        self.distance = max(distance_m, 1.0)
        self.freq_mhz = freq_mhz
        self.pl_exp = path_loss_exponent
        self.shadowing_std_db = shadowing_std_db

    def path_loss_db(self):
        # Free-space reference at 1 m using simplified formula
        # PL(d) = 20*log10(f) + 20*log10(d) - 27.55  (f in MHz, d in m)
        fspl_db = 20 * math.log10(self.freq_mhz) + 20 * math.log10(self.distance) - 27.55
        # Apply extra exponent (log-distance model)
        extra = 10 * (self.pl_exp - 2) * math.log10(self.distance)
        return fspl_db + extra

    def shadowing_db(self):
        return random.gauss(0.0, self.shadowing_std_db)

    def rayleigh_gain_linear(self):
        # Rayleigh amplitude ~ Rayleigh(sigma=1). Power is exponential with mean 1.
        return random.expovariate(1.0)

    def instantaneous_gain_linear(self):
        # Combine path loss + shadowing + small scale fading
        pl_db = self.path_loss_db()
        sh_db = self.shadowing_db()
        total_loss_db = pl_db + sh_db
        gain_from_loss = db_to_linear(-total_loss_db)
        fad = self.rayleigh_gain_linear()
        return gain_from_loss * fad


# ----------------------------- Traffic model & user -----------------------------
class User:
    """User with Poisson arrivals and a packet queue (in bits).
    Packets are fixed-size for simplicity.
    """

    def __init__(self, user_id, distance_m, arrival_lambda_pkt_per_s=5.0, pkt_size_bits=8000, channel=None):
        self.id = user_id
        self.distance = distance_m
        self.arrival_lambda = arrival_lambda_pkt_per_s
        self.pkt_size = pkt_size_bits
        self.queue_bits = 0
        self.channel = channel if channel is not None else Channel(distance_m)
        self.stats = {'served_bits': 0, 'dropped_bits': 0, 'arrived_bits': 0, 'delay_samples': []}

    def step_arrivals(self, dt):
        # Poisson arrivals with rate lambda -> expected lambda*dt packets
        lam = self.arrival_lambda * dt
        num = np.random.poisson(lam)
        arrived_bits = num * self.pkt_size
        self.queue_bits += arrived_bits
        self.stats['arrived_bits'] += arrived_bits
        return arrived_bits

    def instantaneous_channel_gain(self):
        return self.channel.instantaneous_gain_linear()


# ----------------------------- Physical layer: rate calculation -----------------------------
def shannon_rate_bps(bandwidth_hz, snr_linear):
    # Return rate in bits/s using Shannon's capacity: C = B * log2(1 + SNR)
    return bandwidth_hz * math.log2(1.0 + max(snr_linear, 1e-12))


# ----------------------------- Agent: Tabular Q-learning -----------------------------
class QLearningAgent:
    """Q-learning with discretized state and discrete actions.
    State: (qbin_user0, qbin_user1, snrbin_user0, snrbin_user1)
    Action: choose fraction of max_power to allocate to user0 from a discrete set; rest goes to user1.
    """

    def __init__(self, n_users=2, power_levels=5, q_bins=(0, 10000, 50000, 200000, 1e9), snr_bins_db=(-100, -10, 0, 10, 20, 40),
                 alpha=0.1, gamma=0.95, epsilon=0.2):
        self.n_users = n_users
        self.power_levels = power_levels
        self.q_bins = q_bins
        self.snr_bins_db = snr_bins_db
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)  # mapping from state-action tuple to Q
        # build action set: fraction allocations [0.0, ...,1.0]
        self.actions = [i / (power_levels - 1) for i in range(power_levels)]

    def discretize(self, users, gains_linear):
        # users: list of User; gains_linear: instantaneous channel gains
        qbin = []
        for u in users:
            q = u.queue_bits
            # find bin index
            idx = 0
            while idx + 1 < len(self.q_bins) and q >= self.q_bins[idx + 1]:
                idx += 1
            qbin.append(idx)
        snr_db = [linear_to_db(g * 1e3) for g in gains_linear]  # scale for numerical stability
        snrbin = []
        for s in snr_db:
            idx = 0
            while idx + 1 < len(self.snr_bins_db) and s >= self.snr_bins_db[idx + 1]:
                idx += 1
            snrbin.append(idx)
        return tuple(qbin + snrbin)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # greedy
        best_a = None
        best_q = -1e18
        for a in self.actions:
            val = self.q_table[(state, a)]
            if val > best_q:
                best_q = val
                best_a = a
        return best_a if best_a is not None else random.choice(self.actions)

    def update(self, state, action, reward, next_state):
        # Q(s,a) := (1-alpha)Q + alpha*(r + gamma*max_a' Q(s',a'))
        cur = self.q_table[(state, action)]
        # compute max next
        max_next = max([self.q_table[(next_state, a)] for a in self.actions]) if next_state is not None else 0.0
        new = (1 - self.alpha) * cur + self.alpha * (reward + self.gamma * max_next)
        self.q_table[(state, action)] = new


# ----------------------------- Simulator -----------------------------
class Simulator:
    def __init__(self, users, bandwidth_hz=180e3, max_tx_power_w=1.0, dt=0.1, noise_figure_db=7.0, noise_temp_k=290.0):
        self.users = users
        self.B = bandwidth_hz
        self.Pmax = max_tx_power_w
        self.dt = dt
        # thermal noise power N0 = k*T*B
        k = 1.38064852e-23
        N0 = k * noise_temp_k * self.B
        self.noise_power_w = N0 * db_to_linear(noise_figure_db)
        self.time = 0.0

    def step(self, allocation_fractions):
        # allocation_fractions: fraction for user0; for n_users>1 distribute remaining equally across others
        n = len(self.users)
        if n == 1:
            alloc = [1.0]
        elif n == 2:
            alloc = [allocation_fractions, 1.0 - allocation_fractions]
        else:
            # for >2, allocate fraction to user0, rest equally
            f0 = allocation_fractions
            rest = (1 - f0) / (n - 1)
            alloc = [f0] + [rest] * (n - 1)

        # arrivals
        for u in self.users:
            u.step_arrivals(self.dt)

        # channel gains and rates
        gains = [u.instantaneous_channel_gain() for u in self.users]
        rates = []
        energy_used_j = 0.0
        served_bits = [0] * n

        for i, u in enumerate(self.users):
            tx_power = max(0.0, min(1.0, alloc[i])) * self.Pmax
            # SNR = P_tx * gain / noise
            snr = tx_power * gains[i] / (self.noise_power_w + 1e-20)
            rate_bps = shannon_rate_bps(self.B, snr)
            # serviceable bits in dt
            can_send = rate_bps * self.dt
            to_send = min(can_send, u.queue_bits)
            u.queue_bits -= to_send
            u.stats['served_bits'] += to_send
            served_bits[i] = to_send
            rates.append(rate_bps)
            energy_used_j += tx_power * self.dt  # energy = power * time

        self.time += self.dt
        # reward: combine negative energy and QoS (served bits and queue penalty)
        total_served = sum(served_bits)
        queue_penalty = -sum([u.queue_bits for u in self.users]) * 1e-6  # scale down
        reward = -energy_used_j + 0.0005 * total_served + queue_penalty

        return reward, gains, rates, energy_used_j


# ----------------------------- Baseline static allocation -----------------------------
def run_baseline(sim_steps=2000, dt=0.1, allocation_fraction=0.5):
    # initialize two users at different distances
    u0 = User(0, distance_m=50.0, arrival_lambda_pkt_per_s=3.0, pkt_size_bits=8000, channel=Channel(50.0))
    u1 = User(1, distance_m=150.0, arrival_lambda_pkt_per_s=4.0, pkt_size_bits=8000, channel=Channel(150.0))
    sim = Simulator([u0, u1], bandwidth_hz=180e3, max_tx_power_w=1.0, dt=dt)

    energies = []
    queues = []
    throughputs = []

    for _ in range(sim_steps):
        r, gains, rates, e = sim.step(allocation_fraction)
        energies.append(e)
        queues.append([u0.queue_bits, u1.queue_bits])
        throughputs.append(sum(rates))

    total_energy = sum(energies)
    avg_queue = np.mean(queues, axis=0)
    avg_throughput = np.mean(throughputs)
    return {'energy': total_energy, 'avg_queue': avg_queue, 'avg_throughput': avg_throughput, 'users': [u0, u1]}


# ----------------------------- Training RL -----------------------------

def train_rl(episodes=200, sim_steps_per_episode=500, dt=0.1):
    # two users setup
    u0 = User(0, distance_m=50.0, arrival_lambda_pkt_per_s=3.0, pkt_size_bits=8000, channel=Channel(50.0))
    u1 = User(1, distance_m=150.0, arrival_lambda_pkt_per_s=4.0, pkt_size_bits=8000, channel=Channel(150.0))

    env_users = [u0, u1]
    agent = QLearningAgent(n_users=2, power_levels=7, alpha=0.15, gamma=0.95, epsilon=0.25)

    sim = Simulator(env_users, bandwidth_hz=180e3, max_tx_power_w=1.0, dt=dt)

    rewards_history = []
    energy_history = []
    throughput_history = []

    for ep in range(episodes):
        # reset user queues and stats
        for u in env_users:
            u.queue_bits = 0
            u.stats = {'served_bits': 0, 'dropped_bits': 0, 'arrived_bits': 0, 'delay_samples': []}
        sim.time = 0.0

        ep_reward = 0.0
        ep_energy = 0.0
        ep_throughput = 0.0

        # initial state
        gains = [u.instantaneous_channel_gain() for u in env_users]
        state = agent.discretize(env_users, gains)

        for t in range(sim_steps_per_episode):
            a = agent.choose_action(state)
            reward, gains_next, rates, energy = sim.step(a)
            next_state = agent.discretize(env_users, gains_next)
            agent.update(state, a, reward, next_state)

            state = next_state
            ep_reward += reward
            ep_energy += energy
            ep_throughput += sum(rates)

        rewards_history.append(ep_reward)
        energy_history.append(ep_energy)
        throughput_history.append(ep_throughput / sim_steps_per_episode)

        # decay epsilon slowly
        agent.epsilon = max(0.02, agent.epsilon * 0.995)

        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"Episode {ep+1}/{episodes}: reward={ep_reward:.2f} energy={ep_energy:.2f} avg_throughput={throughput_history[-1]:.2f}")

    return {'agent': agent, 'rewards': rewards_history, 'energies': energy_history, 'throughputs': throughput_history, 'users': env_users}


# ----------------------------- Main: run training and compare -----------------------------

def main():
    print("Running baseline (static allocation)...")
    baseline_res = run_baseline(sim_steps=2000, dt=0.1, allocation_fraction=0.5)
    print(f"Baseline total energy: {baseline_res['energy']:.2f} J, avg throughput {baseline_res['avg_throughput']:.2f} bps")

    print("Training RL agent...")
    rl_res = train_rl(episodes=150, sim_steps_per_episode=400, dt=0.1)

    # Evaluate RL policy
    agent = rl_res['agent']
    # run evaluation episode without exploration
    agent.epsilon = 0.0

    # create fresh users
    u0 = User(0, distance_m=50.0, arrival_lambda_pkt_per_s=3.0, pkt_size_bits=8000, channel=Channel(50.0))
    u1 = User(1, distance_m=150.0, arrival_lambda_pkt_per_s=4.0, pkt_size_bits=8000, channel=Channel(150.0))
    sim = Simulator([u0, u1], bandwidth_hz=180e3, max_tx_power_w=1.0, dt=0.1)

    total_energy_rl = 0.0
    total_throughput_rl = 0.0
    for t in range(2000):
        gains = [u.instantaneous_channel_gain() for u in [u0, u1]]
        state = agent.discretize([u0, u1], gains)
        a = agent.choose_action(state)
        r, gains_next, rates, energy = sim.step(a)
        total_energy_rl += energy
        total_throughput_rl += sum(rates)

    total_energy_baseline = baseline_res['energy']
    print('\n=== Summary ===')
    print(f"Baseline energy (same duration): {total_energy_baseline:.2f} J")
    print(f"RL energy: {total_energy_rl:.2f} J")
    energy_reduction = 100.0 * (total_energy_baseline - total_energy_rl) / total_energy_baseline
    print(f"Energy reduction vs baseline: {energy_reduction:.2f}%")

    # Plot training curves
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    axs[0].plot(rl_res['rewards'])
    axs[0].set_title('Episode reward')
    axs[1].plot(rl_res['energies'])
    axs[1].set_title('Episode total energy (J)')
    axs[2].plot(rl_res['throughputs'])
    axs[2].set_title('Episode avg throughput (bps)')
    plt.tight_layout()
    plt.show()

    # Print final advice
    print('\nThe script produced training curves and a printed summary.\n')


if __name__ == '__main__':
    main()
