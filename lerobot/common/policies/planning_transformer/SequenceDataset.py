from torch.utils.data import DataLoader, IterableDataset
import numpy as np


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
        arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(env: OfflineEnv, gamma: float = 1.0, test_size: float = 0.02
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset = env.get_dataset()
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        if "infos/goal" in dataset.keys():
            data_["goals"].append(dataset["infos/goal"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    train_traj, test_traj = train_test_split(traj, test_size=test_size, random_state=42)
    train_obs = np.concatenate([t["observations"] for t in train_traj], axis=0)
    obs_mean, obs_std = train_obs.mean(0, keepdims=True), train_obs.std(0, keepdims=True) + 1e-6

    def get_info(traj_list):
        return {
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "traj_lens": np.array([len(t["actions"]) for t in traj_list]),
        }

    return {"trajectories": train_traj, "info": get_info(train_traj)}, {"trajectories": test_traj,
                                                                        "info": get_info(test_traj)}

class SequencePlanDataset(IterableDataset):
    def __init__(self, dataset, ds_info, seq_len: int = 10, reward_scale: float = 1.0, path_length=10,
                 embedding_dim: int = 128, plan_sampling_method: int = 4, plan_max_trajectory_ratio=0.5,
                 plan_combine: bool = False, plan_disabled: bool = False, plan_indices: Tuple[int, ...] = (0, 1),
                 is_goal_conditioned: bool = False, plans_use_actions: bool = False):
        self.info = ds_info
        self.dataset = dataset
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = self.info["obs_mean"]
        self.state_std = self.info["obs_std"]

        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = self.info["traj_lens"] / self.info["traj_lens"].sum()

        super().__init__(dataset, ds_info, seq_len, reward_scale)
        self.path_length = path_length
        self.plan_indices = range(0, self.state_mean.shape[1]) if plan_indices is None else plan_indices
        self.plans_use_actions = plans_use_actions

        self.is_gc = is_goal_conditioned  # dataset depends on whether reward conditioned (rc) or goal conditioned (gc)
        self.plan_length = (1 if plan_combine else self.path_length) * (not plan_disabled)
        actions_shape = self.dataset[0]["actions"].shape[-1]
        self.path_dim = len(self.plan_indices) + (actions_shape if plans_use_actions else 0)
        self.plan_dim = (self.path_length if plan_combine else 1) \
                        * (not plan_disabled) \
                        * (self.path_dim + (not self.is_gc))
        self.embedding_dim = embedding_dim
        if self.is_gc:
            traj_dists = np.array([self.traj_distance(traj, plan_indices) for traj in self.dataset])
            # self.sample_prob = traj_dists / traj_dists.sum()
            self.sample_prob *= traj_dists / traj_dists.sum()
        self.sample_prob /= self.sample_prob.sum()

        self.expected_cum_reward = np.array(
            [traj['returns'][0] * p for traj, p in zip(self.dataset, self.sample_prob)]
        ).sum()
        self.max_traj_length = max(self.info["traj_lens"])
        self.plan_sampler = PathSampler(method=plan_sampling_method)
        self.plan_max_trajectory_ratio = plan_max_trajectory_ratio

        self.plan_combine = plan_combine

    @staticmethod
    def traj_distance(traj, indices):
        obs = traj['observations'][:, indices]
        return np.linalg.norm(obs[-1] - obs[0])

    def create_plan(self, states, returns=None, actions=None):
        if self.plan_length:
            positions = states[:, self.plan_indices]
            if returns is not None:
                # handle the reward conditioning
                positions = np.hstack((positions, returns[:, np.newaxis]))

            if actions is not None:
                positions = np.hstack((positions, actions))

            path = np.array(self.plan_sampler.sample(positions, self.path_length))
            if self.plan_combine:
                path = pad_along_axis(path[np.newaxis, :], pad_to=self.path_length, axis=1)
                path = path.reshape(-1, self.plan_dim)
            else:
                path = pad_along_axis(path, pad_to=self.plan_length, axis=0)
        else:
            path = np.empty((0, self.plan_dim))

        return path

    def convert_plan_to_path(self, plan, plan_path_viz_indices):
        if self.plan_combine:
            plan = plan[0].reshape(*plan[0].shape[:-1], -1, self.path_dim)

        return plan[:, plan_path_viz_indices]

    def _prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa

        # Hindsight goal relabelling only if the goal was actually achieved
        # Relabelling all trajectories can cause it to learn that bad actions still reach the goal
        # if traj["returns"][0:1]>0: goal = traj["observations"][-1:, :2]

        states = traj["observations"][start_idx: start_idx + self.seq_len]
        states_till_end = traj["observations"][start_idx:]

        # create the plan from the current state minus some random amount to at most half the max episode length
        # we subtract this random amount to make sure eval doesn't go OOD when the start state doesn't match.
        plan_states_start = max(0, start_idx + random.randint(-self.seq_len, 0))
        plan_states_end = start_idx + (
            max(math.floor(random.uniform(0.5, 1.0) * len(states_till_end)), self.path_length)
            if self.is_gc
            else int(self.max_traj_length * self.plan_max_trajectory_ratio)) + 1
        # plan_states_end = len(traj["observations"])
        plan_states = traj["observations"][plan_states_start:plan_states_end]
        plan_returns = traj["returns"][plan_states_start:plan_states_end] * self.reward_scale

        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        returns = traj["returns"][start_idx: start_idx + self.seq_len] * self.reward_scale
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = normalize_state(states, self.state_mean, self.state_std)

        states_till_end = normalize_state(states_till_end, self.state_mean, self.state_std)
        plan_states = normalize_state(plan_states, self.state_mean, self.state_std)

        if self.is_gc:
            # select random observation in future
            # since the plan already implements this logic we just select the last plan state
            goal = plan_states[-1:]
            # if "goals" in traj.keys():
            #     # ant maze specific fix in future
            #     goal = traj["goals"][0:1].astype(np.float32)
            #     goal = normalize_state(goal, self.state_mean[0:1, :2], self.state_std[0:1, :2])
            # else:
            #     # for other environments select random observation in future
            #     # since the plan already implements this logic we just select the last plan state
            #     goal = plan_states[-1:]
        else:
            goal = np.zeros((1, 1), dtype=np.float32)

        plan_states = plan_states[:int(self.max_traj_length * self.plan_max_trajectory_ratio)]
        plan_actions = traj["actions"][plan_states_start:plan_states_start + len(plan_states)] \
            if self.plans_use_actions else None
        if self.is_gc:
            plan = self.create_plan(plan_states, actions=plan_actions).astype(np.float32)
        else:
            plan_returns = plan_returns[:int(self.max_traj_length * self.plan_max_trajectory_ratio)]
            plan = self.create_plan(plan_states, returns=plan_returns, actions=plan_actions).astype(np.float32)

        # pad up to seq_len if needed, padding is masked during training
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        steps_till_end = states_till_end.shape[0]
        if steps_till_end < self.max_traj_length:
            states_till_end = pad_along_axis(states_till_end, pad_to=self.max_traj_length)

        weight = (traj["returns"][0] + self.reward_scale) / (self.expected_cum_reward + self.reward_scale)

        return goal, states, actions, returns, time_steps, mask, plan, states_till_end, steps_till_end, weight

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self._prepare_sample(traj_idx, start_idx)


def construct_sequence_with_goal_and_plan(goal, plan, rsa_sequence):
    first_rs = rsa_sequence[:, :2]  # shape [batch_size, 2, emb_dim (or 1 if mask)]
    remaining_elements = rsa_sequence[:, 2:]  # shape [batch_size, 3*seq_len-2, emb_dim (or 1 if mask)]
    return torch.cat([goal, first_rs, plan, remaining_elements], dim=1)


def un_normalise_state(state, state_mean, state_std):
    return (state * state_std) + state_mean