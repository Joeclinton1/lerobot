"""
Modified version of the single file implementation of Decision transformer as provided by the CORL team
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
            self,
            seq_len: int,
            embedding_dim: int,
            num_heads: int,
            attention_dropout: float,
            residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
            self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, log_attention: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        norm_x = self.norm1(x)

        attention = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=log_attention,
            average_attn_weights=False,
            is_causal=False,
        )
        attention_out = attention[0]

        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x, attention[1] if log_attention else None


class PlanningDecisionTransformer(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 plan_dim: int,
                 seq_len: int = 10,
                 embedding_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 attention_dropout: float = 0.0,
                 residual_dropout: float = 0.0,
                 plan_length: int = 1,
                 use_two_phase_training: bool = False,
                 goal_indices: Tuple[int, ...] = (0, 1),
                 plan_indices: Tuple[int, ...] = (0, 1),
                 non_plan_downweighting: float = 0.0,
                 use_timestep_embedding: bool = True,
                 plan_use_relative_states: bool = True,
                 goal_representation: int = 3,
                 embedding_dropout: float = 0.0,
                 episode_len: int = 1000,
                 max_action: float = 1.0,
                 ):
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps

        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)
        self.goal_cond = len(goal_indices) > 0
        self.goal_length = 0 if not self.goal_cond else 2 if goal_representation in [3, 4] else 1

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len + plan_length + self.goal_length,
                    # Adjusted for the planning token and goal
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.plan_length = plan_length
        self.non_plan_downweighting = non_plan_downweighting
        self.planning_head = nn.Linear(embedding_dim, plan_dim)
        self.plan_emb = nn.Linear(plan_dim, embedding_dim)
        self.goal_emb = nn.Linear(len(goal_indices), embedding_dim)
        self.plan_positional_emb = nn.Embedding(plan_length, embedding_dim)
        self.full_seq_pos_emb = nn.Embedding(3 + plan_length, embedding_dim)
        self.plan_dim = plan_dim
        self.use_two_phase_training = use_two_phase_training
        self.use_timestep = use_timestep_embedding
        self.goal_indices = torch.tensor(goal_indices, dtype=torch.long)
        self.plan_indices = torch.tensor(plan_indices, dtype=torch.long)

        # Create position IDs for plan once
        self.register_buffer('plan_position_ids', torch.arange(0, self.plan_length).unsqueeze(0))
        self.register_buffer('full_seq_pos_ids',
                             torch.arange(0,
                                          3 * seq_len + plan_length + self.goal_length).unsqueeze(
                                 0))

        # increase focus on plan by downweighting non plan tokens
        self.original_causal_masks = [block.causal_mask for block in self.blocks]

        self.apply(self._init_weights)

        # ablation testing variables for plan/goal representation
        self.plan_use_relative_states = plan_use_relative_states
        self.goal_representation = goal_representation

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @staticmethod
    def construct_sequence_with_goal_and_plan(goal, plan, rsa_sequence):
        first_rs = rsa_sequence[:, :2]  # shape [batch_size, 2, emb_dim (or 1 if mask)]
        remaining_elements = rsa_sequence[:, 2:]  # shape [batch_size, 3*seq_len-2, emb_dim (or 1 if mask)]
        return torch.cat([goal, first_rs, plan, remaining_elements], dim=1)

    def downweight_non_plan(self, plan_start, plan_length, downweighting):
        for i, block in enumerate(self.blocks):
            # attention takes a causal mask which is 0.0 if we fully attend and -inf to avoid attending
            # however by providing a value between -inf and 0.0 for the columns which are not the plans, we effectively
            # downweight the tokens attention towards non plan tokens, helping focus more on the plans
            new_attn_mask = torch.full(block.causal_mask.shape, downweighting, dtype=torch.float32)
            new_attn_mask[0:plan_start + plan_length, :] = 0.0
            new_attn_mask[:, plan_start:plan_start + plan_length] = 0.0
            new_attn_mask.masked_fill_(self.original_causal_masks[i], float('-inf')).fill_diagonal_(0.0)
            block.causal_mask = new_attn_mask.to(block.causal_mask.device)

    def forward(self, goal, states, actions, returns_to_go, time_steps, plan, padding_mask=None,
                log_attention=False):

        batch_size, seq_len = states.shape[0], states.shape[1]
        device = states.device
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        # print(states.shape, self.state_emb.in_features, self.state_emb.out_features)
        state_emb_no_time_emb = self.state_emb(states)
        # state_emb_no_time_emb = self.plan_emb(states[:, :, :2])
        state_emb = state_emb_no_time_emb + (time_emb if self.use_timestep else 0)
        act_emb = self.action_emb(actions) + (time_emb if self.use_timestep else 0)
        # remove action conditioning (would this help?)
        # act_emb = torch.zeros (size=act_emb.shape, dtype=torch.float32, device=act_emb.device)
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + (time_emb if self.use_timestep else 0)
        for training_phase in range(self.use_two_phase_training + 1):
            plan_pos_emb = self.plan_positional_emb(self.plan_position_ids[:, :plan.shape[1]])
            # make plan relative, accounting for the possibility of actions in plan
            # can also add pos_emb here if don't want the full sequence embedding
            if self.plan_length:
                plan_states = plan[:, :, :len(self.plan_indices)].clone()
                if self.plan_use_relative_states:
                    plan_states -= states[:, :1, self.plan_indices]

                plan_emb = self.plan_emb(torch.cat((
                    plan_states,
                    plan[:, :, len(self.plan_indices):]
                ), dim=-1)) + plan_pos_emb

            else:
                plan_emb = torch.empty(batch_size, 0, self.embedding_dim, device=device)
            # plan_emb = self.plan_emb(plan) + plan_pos_emb if self.plan_length else \
            #     torch.empty(batch_size, 0, self.embedding_dim, device=device)

            # handle goal
            # we do this inserting the goal into the state, embedding it then subtracting the state embedding
            # we detatch the state embedding to prevent co-dependency during backprop
            # goal_modified_state_0 = states[:, 0:1].clone().detach()
            # goal_modified_state_0[:, :, self.goal_indices] = goal[:, :, self.goal_indices]
            # goal_token = (self.state_emb(goal_modified_state_0) - state_emb_no_time_emb[:, 0:1, :]).detach()
            # goal_token = torch.zeros(state_emb.shape, dtype=torch.float32, device=goal.device)[:,:1]
            # goal_token[:,:1,:2]=goal[:, :1, self.goal_indices] - states[:, :1, self.goal_indices]
            if self.goal_cond:
                if self.goal_representation == 1:
                    # absolute goal
                    goal_token = self.goal_emb(goal[:, :1, self.goal_indices])
                elif self.goal_representation == 2:
                    # relative goal
                    goal_token = self.goal_emb(goal[:, :1, self.goal_indices] - states[:, :1, self.goal_indices])
                elif self.goal_representation == 3:
                    # goal space state + absolute goal
                    goal_token = self.goal_emb(
                        torch.cat((states[:, :1, self.goal_indices],
                                   goal[:, :1, self.goal_indices]), dim=1))
                else:
                    # goal space state + relative goal
                    goal_token = self.goal_emb(
                        torch.cat((states[:, :1, self.goal_indices],
                                   goal[:, :1, self.goal_indices] - states[:, :1, self.goal_indices]), dim=1))
            else:
                goal_token = torch.empty(batch_size, 0, self.embedding_dim, device=device)

            # if(batch_size == 1):
            #     print(goal_token)
            # goal_token = self.goal_emb(goal[:, :1, self.goal_indices])

            # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
            sequence = (
                torch.stack([returns_emb, state_emb, act_emb], dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, 3 * seq_len, self.embedding_dim)
            )
            # convert to form (goal, r_0, s_0, p_0, p1, ..., p_n, a_0, r_1, s_1, a_1, ...)
            sequence = self.construct_sequence_with_goal_and_plan(goal_token, plan_emb, sequence)
            # sequence[:, :2 + plan.shape[1]] += self.full_seq_pos_emb(self.full_seq_pos_ids[:, :2 + plan.shape[1]])
            padding_mask_full = None
            if padding_mask is not None:
                # [batch_size, seq_len * 3], stack mask identically to fit the sequence
                padding_mask_full = (
                    torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                    .permute(0, 2, 1)
                    .reshape(batch_size, 3 * seq_len)
                )
                # account for the planning token in the mask
                # True values in the mask mean don't attend to, so we use zeroes so the plan and goal are always attended to
                plan_mask = torch.zeros(plan_emb.shape[:2], dtype=torch.bool,
                                        device=device)
                goal_mask = torch.zeros(goal_token.shape[:2], dtype=torch.bool,
                                        device=device)

                padding_mask_full = self.construct_sequence_with_goal_and_plan(goal_mask, plan_mask, padding_mask_full)

            # LayerNorm and Dropout (!!!) as in original implementation,
            # while minGPT & huggingface uses only embedding dropout
            out = self.emb_norm(sequence)
            out = self.emb_drop(out)

            # for some interpretability lets get the attention maps
            attention_maps = []
            for i, block in enumerate(self.blocks):
                out, attn_weights = block(out, padding_mask=padding_mask_full, log_attention=log_attention)
                attention_maps.append(attn_weights)

            out = self.out_norm(out)

            start = 1 + self.goal_length
            if training_phase == 0:
                # for input to the planning_head we use the sequence shifted one to the left of the plan_sequence
                plan = self.planning_head(out[:, start: start + self.plan_length])
            if training_phase == self.use_two_phase_training:
                # predict actions only from the state embeddings
                out_states = torch.cat([out[:, start:start + 1], out[:, (start + 3 + self.plan_length)::3]], dim=1)
                out_actions = self.action_head(out_states) * self.max_action  # [batch_size, seq_len, action_dim]
        return plan, out_actions, attention_maps

