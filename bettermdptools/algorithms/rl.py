"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE

modified by: John Mansfield

documentation added by: Gagandeep Randhawa

Class that contains functions related to reinforcement learning algorithms. RL init expects an OpenAI environment (env).

Model-free learning algorithms: Q-Learning and SARSA
work out of the box with any gymnasium environments that 
have single discrete valued state spaces, like frozen lake. A lambda function 
is required to convert state spaces not in this format.
"""

import warnings
from gymnasium import Env
from typing import Dict, List, Tuple, Callable, Any
import numpy as np
from tqdm import tqdm
from bettermdptools.utils.callbacks import MyCallbacks


class RL:
    """
    Class that contains functions related to reinforcement learning algorithms. RL init expects an OpenAI environment (env).

    Model-free learning algorithms: Q-Learning and SARSA
    work out of the box with any gymnasium environments that
    have single discrete valued state spaces, like frozen lake. A lambda function
    is required to convert state spaces not in this format.
    """

    def __init__(self, env: Env) -> None:
        self.env = env
        self.callbacks = MyCallbacks()
        self.render = False
        # Explanation of lambda:
        # def select_action(state, Q, epsilon):
        #   if np.random.random() > epsilon:
        #       max_val = np.max(Q[state])
        #       indxs_selector = np.isclose(Q[state], max_val)
        #       indxs = np.arange(len(Q[state]))[indxs_selector]
        #       return np.random.choice(indxs)
        #   else:
        #       return np.random.randint(len(Q[state]))
        self.select_action: Callable[[int, np.ndarray, float], int] = (
            lambda state, Q, epsilon: (
                np.random.choice(
                    np.arange(len(Q[state]))[np.isclose(Q[state], np.max(Q[state]))]
                )
                if np.random.random() > epsilon
                else np.random.randint(len(Q[state]))
            )
        )

    @staticmethod
    def decay_schedule(
        init_value: float,
        min_value: float,
        decay_ratio: float,
        max_steps: int,
        log_start: float = -2,
        log_base: float = 10,
    ) -> np.ndarray:
        """
        Parameters
        ----------------------------
        init_value {float}:
            Initial value of the quantity being decayed

        min_value {float}:
            Minimum value init_value is allowed to decay to

        decay_ratio {float}:
            The exponential factor exp(decay_ratio).
            Updated decayed value is calculated as

        max_steps {int}:
            Max iteration steps for decaying init_value

        log_start {array-like}, default = -2:
            Starting value of the decay sequence.
            Default value starts it at 0.01

        log_base {array-like}, default = 10:
            Base of the log space.


        Returns
        ----------------------------
        values {array-like}, shape(max_steps):
            Decay values where values[i] is the value used at i-th step
        """
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[
            ::-1
        ]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")
        return values

    def q_learning(
        self,
        nS: int | None = None,
        nA: int | None = None,
        convert_state_obs: Callable[[Any], int] = lambda state: state,
        gamma: float = 0.99,
        init_alpha: float = 0.5,
        min_alpha: float = 0.01,
        alpha_decay_ratio: float = 0.5,
        init_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        epsilon_decay_ratio: float = 0.9,
        n_episodes: int = 10000,
        verbose: bool = False,
        include_reward_history: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], np.ndarray, List[np.ndarray]]:
        """
        Parameters
        ----------------------------
        nS {int}:
            Number of states

        nA {int}:
            Number of available actions

        convert_state_obs {lambda}:
            Converts state into an integer

        gamma {float}, default = 0.99:
            Discount factor

        init_alpha {float}, default = 0.5:
            Learning rate

        min_alpha {float}, default = 0.01:
            Minimum learning rate

        alpha_decay_ratio {float}, default = 0.5:
            Decay schedule of learing rate for future iterations

        init_epsilon {float}, default = 1.0:
            Initial epsilon value for epsilon greedy strategy.
            Chooses max(Q) over available actions with probability 1-epsilon.

        min_epsilon {float}, default = 0.1:
            Minimum epsilon. Used to balance exploration in later stages.

        epsilon_decay_ratio {float}, default = 0.9:
            Decay schedule of epsilon for future iterations

        n_episodes {int}, default = 10000:
            Number of episodes for the agent

        verbose {bool}, default = False:
            Whether to print progress to the console

        include_reward_history {bool}, default = False:
            Whether to include reward history in the output

        Returns
        ----------------------------
        Q {numpy array}, shape(nS, nA):
            Final action-value function Q(s,a)

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.

        V {numpy array}, shape(nS):
            State values array

        Q_track {numpy array}, shape(n_episodes, nS, nA):
            Log of Q(s,a) for each episode

        pi_track {list}, len(n_episodes):
            Log of complete policy for each episode

        reward_history {np.ndarray}, shape(n_episodes):
            Log of reward history for each episode
        """
        if nS is None:
            nS = self.env.observation_space.n
        if nA is None:
            nA = self.env.action_space.n
        pi_track: List[np.ndarray] = []
        reward_history: np.ndarray = np.zeros(
            n_episodes, dtype=np.float64
        )  # Changed to float64
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        alphas = RL.decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
        epsilons = RL.decay_schedule(
            init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
        )

        for e in tqdm(range(n_episodes), leave=False, disable=verbose):
            if verbose:
                print(f"Q-Learner Episode {e}")
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            state, info = self.env.reset()
            done = False
            state = convert_state_obs(state)

            # Track episode reward
            episode_reward = 0.0

            while not done:
                if self.render:
                    warnings.warn(
                        "Occasional render has been deprecated by openAI.  Use test_env.py to render."
                    )
                action = self.select_action(state, Q, epsilons[e])
                if verbose:
                    print(f"Q-Learner Stepping with action {action}")
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Accumulate reward for this episode
                episode_reward += reward

                if truncated:
                    warnings.warn(
                        "Episode was truncated.  TD target value may be incorrect."
                    )
                done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state)
                td_target = reward + gamma * Q[next_state].max() * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state = next_state
                if verbose:
                    print("Q-Learner Updated Q table and state.")

            # Store total episode reward
            if include_reward_history:
                reward_history[e] = episode_reward

            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V: np.ndarray = np.max(Q, axis=1)
        pi: Dict[int, int] = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}

        if include_reward_history:
            return Q, V, pi, Q_track, pi_track, reward_history
        else:
            return Q, V, pi, Q_track, pi_track

    def sarsa(
        self,
        nS: int = None,
        nA: int = None,
        convert_state_obs: Callable[[Any], int] = lambda state: state,
        gamma: float = 0.99,
        init_alpha: float = 0.5,
        min_alpha: float = 0.01,
        alpha_decay_ratio: float = 0.5,
        init_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        epsilon_decay_ratio: float = 0.9,
        n_episodes: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], np.ndarray, List[np.ndarray]]:
        """
        Parameters
        ----------------------------
        nS {int}:
            Number of states

        nA {int}:
            Number of available actions

        convert_state_obs {lambda}:
            Converts state into an integer

        gamma {float}, default = 0.99:
            Discount factor

        init_alpha {float}, default = 0.5:
            Learning rate

        min_alpha {float}, default = 0.01:
            Minimum learning rate

        alpha_decay_ratio {float}, default = 0.5:
            Decay schedule of learing rate for future iterations

        init_epsilon {float}, default = 1.0:
            Initial epsilon value for epsilon greedy strategy.
            Chooses max(Q) over available actions with probability 1-epsilon.

        min_epsilon {float}, default = 0.1:
            Minimum epsilon. Used to balance exploration in later stages.

        epsilon_decay_ratio {float}, default = 0.9:
            Decay schedule of epsilon for future iterations

        n_episodes {int}, default = 10000:
            Number of episodes for the agent


        Returns
        ----------------------------
        Q {numpy array}, shape(nS, nA):
            Final action-value function Q(s,a)

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.

        V {numpy array}, shape(nS):
            State values array

        Q_track {numpy array}, shape(n_episodes, nS, nA):
            Log of Q(s,a) for each episode

        pi_track {list}, len(n_episodes):
            Log of complete policy for each episode
        """
        if nS is None:
            nS = self.env.observation_space.n
        if nA is None:
            nA = self.env.action_space.n
        pi_track: List[np.ndarray] = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        alphas = RL.decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
        epsilons = RL.decay_schedule(
            init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes
        )

        for e in tqdm(range(n_episodes), leave=False):
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            state, info = self.env.reset()
            done = False
            state = convert_state_obs(state)
            action = self.select_action(state, Q, epsilons[e])
            while not done:
                if self.render:
                    warnings.warn(
                        "Occasional render has been deprecated by openAI.  Use test_env.py to render."
                    )
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if truncated:
                    warnings.warn(
                        "Episode was truncated.  TD target value may be incorrect."
                    )
                done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state)
                next_action = self.select_action(next_state, Q, epsilons[e])
                td_target = reward + gamma * Q[next_state][next_action] * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state, action = next_state, next_action
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V = np.max(Q, axis=1)

        pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return Q, V, pi, Q_track, pi_track
