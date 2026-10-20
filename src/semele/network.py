"""Who can see whom: the information graph between batteries.

The simulator only reads each agent's neighbour list.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Agent:
    id: int
    neighbors: list[int] = field(default_factory=list)


@dataclass
class Network:
    agents: dict[int, Agent] = field(default_factory=dict)

    def add_edge(self, a: int, b: int) -> None:
        """Undirected edge: a sees b and b sees a."""
        if b not in self.agents[a].neighbors:
            self.agents[a].neighbors.append(b)
        if a not in self.agents[b].neighbors:
            self.agents[b].neighbors.append(a)

    def laplacian(self) -> np.ndarray:
        """Graph Laplacian L = D - A (the operator in the paper's Lemma 1)."""
        n = len(self.agents)
        A = np.zeros((n, n))
        for i, agent in self.agents.items():
            for j in agent.neighbors:
                A[i, j] = 1.0
        return np.diag(A.sum(axis=1)) - A


def _empty(n: int) -> Network:
    return Network(agents={i: Agent(id=i) for i in range(n)})


def make_linear(n: int) -> Network:
    """Chain 0 - 1 - 2 - ... - n-1."""
    net = _empty(n)
    for i in range(n - 1):
        net.add_edge(i, i + 1)
    return net


def make_star(n: int) -> Network:
    """Agent 0 is the hub, everyone else links only to it."""
    net = _empty(n)
    for i in range(1, n):
        net.add_edge(0, i)
    return net


def make_small_world(n: int, k_nn: int = 4, p: float = 0.1, seed: int = 42) -> Network:
    """Watts-Strogatz: ring with k_nn nearest neighbours, each edge rewired with probability p."""
    rng = random.Random(seed)
    net = _empty(n)
    ring = [(i, (i + j) % n) for i in range(n) for j in range(1, k_nn // 2 + 1)]
    for i, target in ring:
        if rng.random() < p:
            new = rng.randint(0, n - 1)
            tries = 0
            while (new == i or new in net.agents[i].neighbors) and tries < n:
                new = rng.randint(0, n - 1)
                tries += 1
            if new != i and new not in net.agents[i].neighbors:
                net.add_edge(i, new)
        elif target not in net.agents[i].neighbors:
            net.add_edge(i, target)
    return net
