from collections import OrderedDict

import torch
import torch.autograd as autograd
import torch.utils.checkpoint as cp
from torch import nn

from .encoders.dlinear import DLinearModel
from .modules import get_child_dict, Module, BatchNorm2d


def make(args):
    """
    Initializes a random meta model.
    """
    model = MAML(args)
    return model


def load(ckpt, args):
    """
    Initializes a meta model with a pre-trained encoder.
    """
    model = MAML(args, ckpt)
    return model


class MAML(Module):
    def __init__(self, args, ckpt=None):
        super(MAML, self).__init__()
        self.args = args
        self.encoder = DLinearModel(args)
        if ckpt is not None:
            self.encoder.load_state_dict(ckpt['encoder_state_dict'])

    def _inner_forward(self, x, params, episode):
        """ Forward pass for the inner loop. """
        feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
        return feat

    def _inner_iter(self, x, y, params, mom_buffer, episode, detach):
        """
        Performs one inner-loop iteration of MAML including the forward and
        backward passes and the parameter update.

        Args:
          x (float tensor, [n_way * n_shot, H, D]): per-episode support set.
          y (int tensor, [n_way * n_shot, P, D]): per-episode support set labels.
          params (dict): the model parameters BEFORE the update.
          mom_buffer (dict): the momentum buffer BEFORE the update.
          episode (int): the current episode index.
          detach (bool): if True, detachs the graph for the current iteration.

        Returns:
          updated_params (dict): the model parameters AFTER the update.
          mom_buffer (dict): the momentum buffer AFTER the update.
        """
        loss_fn = nn.MSELoss()
        with torch.enable_grad():
            # forward pass
            logits = self._inner_forward(x, params, episode)

            f_dim = -1 if self.args.features == 'MS' else 0
            logits = logits[..., f_dim]
            y = y[..., f_dim]

            loss = loss_fn(logits, y)
            # backward pass
            # 不调用backward，而是直接计算loss关于params的梯度
            # 这样的话，模型的参数暂时不会更新，所有任务都会从同一个初始参数开始进行内环更新
            grads = autograd.grad(loss, params.values(),
                                  create_graph=not detach,
                                  only_inputs=True, allow_unused=True)
            # parameter update
            updated_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                if grad is None:
                    updated_param = param
                else:
                    updated_param = param - self.args.learning_rate * grad
                if detach:
                    updated_param = updated_param.detach().requires_grad_(True)
                updated_params[name] = updated_param

        return updated_params, mom_buffer

    def _adapt(self, x, y, params, episode, meta_train):
        """
        Performs inner-loop adaptation in MAML.

        Args:
          x (float tensor, [n_way * n_shot, H, D]): per-episode support set.
            (T: transforms, C: channels, H: height, W: width)
          y (int tensor, [n_way * n_shot, P, D]): per-episode support set labels.
          params (dict): a dictionary of parameters at meta-initialization.
          episode (int): the current episode index.
          meta_train (bool): if True, the model is in meta-training.

        Returns:
          params (dict): model paramters AFTER inner-loop adaptation.
        """
        assert x.dim() == 3 and y.dim() == 3
        assert x.size(0) == y.size(0)

        # Initializes a dictionary of momentum buffer for gradient descent in the
        # inner loop. It has the same set of keys as the parameter dictionary.
        mom_buffer = OrderedDict()
        params_keys = tuple(params.keys())
        mom_buffer_keys = tuple(mom_buffer.keys())

        for m in self.modules():
            if isinstance(m, BatchNorm2d) and m.is_episodic():
                m.reset_episodic_running_stats(episode)

        # 循环进行多次内环更新并返回更新后的参数
        for step in range(self.args.n_step):
            if self.efficient:  # checkpointing
                def _inner_iter_cp(episode, *state):
                    """
                    Performs one inner-loop iteration when checkpointing is enabled.
                    The code is executed twice:
                      - 1st time with torch.no_grad() for creating checkpoints.
                      - 2nd time with torch.enable_grad() for computing gradients.
                    """
                    params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
                    mom_buffer = OrderedDict(zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

                    detach = not torch.is_grad_enabled()  # detach graph in the first pass
                    self.is_first_pass(detach)
                    params, mom_buffer = self._inner_iter(x, y, params, mom_buffer, int(episode), detach)
                    state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
                                  for t in tuple(params.values()) + tuple(mom_buffer.values()))
                    return state

                state = tuple(params.values()) + tuple(mom_buffer.values())
                state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
                params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
                mom_buffer = OrderedDict(zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
            else:
                params, mom_buffer = self._inner_iter(x, y, params, mom_buffer, episode, not meta_train)

        return params

    def forward(self, x_shot, x_query, y_shot, meta_train):
        """
        Args:
          x_shot (float tensor, [n_episode, n_way * n_shot, H, D]): support sets.
          x_query (float tensor, [n_episode, n_way * n_query, P, D]): query sets.
            (T: transforms, C: channels, H: height, W: width)
          y_shot (int tensor, [n_episode, n_way * n_shot, H, D]): support set labels.
          meta_train (bool): if True, the model is in meta-training.

        Returns:
          y_query (float tensor, [n_episode, n_way * n_shot, P, D]): predicted logits.
        """
        assert self.encoder is not None
        assert x_shot.dim() == 4 and x_query.dim() == 4
        assert x_shot.size(0) == x_query.size(0)

        # a dictionary of parameters that will be updated in the inner loop
        # 这些数据是新复制的一份，不会影响模型本身的参数
        params = OrderedDict(self.named_parameters())
        # Frozen if needed
        # for name in list(params.keys()):
        #     if not params[name].requires_grad or any(s in name for s in self.args.frozen + ['temp']):
        #         params.pop(name)

        y_query = []

        # 对于每一个task，进行内环更新并计算查询集上的表现
        # 对于每一个任务，它们初始化的参数都是相同的（即meta-learnt的初始参数，对应于params）
        # 然后通过支持集进行若干次梯度更新，得到更新后的参数
        for ep in range(x_shot.size(0)):
            # inner-loop training: 更新一次参数
            self.train()
            if not meta_train:
                for m in self.modules():
                    if isinstance(m, BatchNorm2d) and not m.is_episodic():
                        m.eval()

            updated_params = self._adapt(x_shot[ep], y_shot[ep], params, ep, meta_train)

            # inner-loop validation: 更新参数之后获得在查询集上的输出，返回并给外部用于计算loss
            with torch.set_grad_enabled(meta_train):
                self.eval()
                logits_ep = self._inner_forward(x_query[ep], updated_params, ep)

            y_query.append(logits_ep)

        self.train(meta_train)
        y_query = torch.stack(y_query)

        return y_query
