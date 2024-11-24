"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE

modified by: John Mansfield

documentation added by: Gagandeep Randhawa

Class that contains functions related to planning algorithms (Value Iteration, Policy Iteration). 
Planner init expects a reward and transitions matrix P, which is a nested dictionary gym style discrete environment 
where P[state][action] is a list of tuples (probability, next state, reward, terminal).

Model-based learning algorithms: Value Iteration and Policy Iteration
"""

from typing import Dict, List, Tuple
import warnings
import numpy as np


class Planner:
    """
    Class that contains functions related to planning algorithms (Value Iteration, Policy Iteration).
    Planner init expects a reward and transitions matrix P, which is a nested dictionary gym style discrete environment
    where P[state][action] is a list of tuples (probability, next state, reward, terminal).
    """

    def __init__(
        self, P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]
    ) -> None:
        self.P = P

    def value_iteration(
        self,
        gamma: float = 1.0,
        n_iters: int = 1000,
        theta: float = 1e-10,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for value iteration.
            State values are considered to be converged when the maximum difference between new and previous state values is less than theta.
            Stops at n_iters or theta convergence - whichever comes first.

        verbose {bool}, default = False:
            Whether to print progress to the console

        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {Dict[int, int]}:
            Policy mapping states to actions.
        """
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            if verbose:
                print(f"Value Iteration Iteration {i}")
            i += 1
            Q = -1 * np.ones(
                (len(self.P), len(next(iter(self.P.values())))), dtype=np.float64
            )
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                converged = True
            V = np.max(Q, axis=1)
            V_track[i] = V
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")

        pi = {s: int(a) for s, a in enumerate(np.argmax(Q, axis=1))}
        return V, V_track, pi

    def policy_iteration(
        self,
        gamma: float = 1.0,
        n_iters: int = 50,
        theta: float = 1e-10,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for policy evaluation.
            State values are considered to be converged when the maximum difference between new and previous state
            values is less than theta.

        verbose {bool}, default = False:
            Whether to print progress to the console

        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {Dict[int, int]}:
            Policy mapping states to actions.
        """
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))

        pi: Dict[int, int] = {s: a for s, a in enumerate(random_actions)}
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            if verbose:
                print(f"Policy Iteration Iteration {i}")
            i += 1
            old_pi = pi
            V = self.policy_evaluation(pi, V, gamma, theta)
            V_track[i] = V
            pi = self.policy_improvement(V, gamma)
            if old_pi == pi:
                converged = True
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")
        return V, V_track, pi

    def policy_evaluation(
        self,
        pi: Dict[int, int],
        prev_V: np.ndarray,
        gamma: float = 1.0,
        theta: float = 1e-10,
    ) -> np.ndarray:
        """
        PARAMETERS:

        pi {Dict[int, int]}:
            Policy mapping states to actions.

        prev_V {numpy array}, shape(possible states):
            Previous state values array

        RETURNS:
            V {numpy array}, shape(possible states):
                Updated state values array
        """
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                a = pi[s]
                for prob, next_state, reward, done in self.P[s][a]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        return V

    def policy_improvement(self, V: np.ndarray, gamma: float = 1.0) -> Dict[int, int]:
        """
        PARAMETERS:

        V {numpy array}, shape(possible states):
            State values array

        RETURNS:
            new_pi {Dict[int, int]}:
                Updated policy mapping states to actions.
        """
        Q = -1 * np.ones(
            (len(self.P), len(next(iter(self.P.values())))), dtype=np.float64
        )
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        new_pi: Dict[int, int] = {s: int(a) for s, a in enumerate(np.argmax(Q, axis=1))}
        return new_pi
