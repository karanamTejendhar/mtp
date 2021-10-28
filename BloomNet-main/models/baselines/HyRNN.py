import torch
import torch.nn as nn
import geoopt.manifolds.poincare.math as pmath
import geoopt

# from hyrnn.lookup_embedding import LookupEmbedding
# from hyrnn.nets import MobiusGRU

import functools


import torch
from torch.nn.modules.module import Module
import geoopt


class LookupEmbedding(Module):
    r"""A lookup table for embeddings, similar to :meth:`torch.nn.Embedding`,
    that replaces operations with their Poincare-ball counterparts.

    This module is intended to be used for word embeddings,
    retrieved by their indices.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim
        (int or tuple of ints): the shape of each embedding;
                                would've been better named embedding_shape,
                                if not for desirable name-level compatibility
                                with nn.Embedding;
                                embedding is commonly a vector,
                                but we do not impose such restriction
                                so as to not prohibit e.g. Stiefel embeddings.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, *embedding_dim).

    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    """

    def __init__(
        self, num_embeddings, embedding_dim, manifold=geoopt.Euclidean(), _weight=None
    ):
        super(LookupEmbedding, self).__init__()
        if isinstance(embedding_dim, int):
            embedding_dim = (embedding_dim,)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold

        if _weight is None:
            _weight = torch.Tensor(num_embeddings, *embedding_dim)
            self.weight = geoopt.ManifoldParameter(_weight, manifold=self.manifold)
            self.reset_parameters()
        else:
            assert _weight.shape == (
                num_embeddings,
                *embedding_dim,
            ), "_weight MUST be of shape (num_embeddings, *embedding_dim)"
            self.weight = geoopt.ManifoldParameter(_weight, manifold=self.manifold)

    def reset_parameters(self):
        # TODO: allow some sort of InitPolicy
        #       as LookupEmbedding's parameter
        #       for e.g. random init;
        #       at the moment, you're supposed
        #       to do actual init on your own
        #       in the client code.
        with torch.no_grad():
            self.weight.fill_(0)

    def forward(self, input):
        shape = list(input.shape) + list(self.weight.shape[1:])
        shape = tuple(shape)
        return self.weight.index_select(0, input.reshape(-1)).view(shape)

import itertools
import torch.nn
import torch.nn.functional
import math
import geoopt.manifolds.poincare.math as pmath
import geoopt


def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    c=1.0,
):
    if hyperbolic_input:
        output = pmath.mobius_matvec(weight, input, c=c)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, c=c)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, c=c)
        output = pmath.mobius_add(output, bias, c=c)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, c=c)
    output = pmath.project(output, c=c)
    return output


def one_rnn_transform(W, h, U, x, b, c):
    W_otimes_h = pmath.mobius_matvec(W, h, c=c)
    U_otimes_x = pmath.mobius_matvec(U, x, c=c)
    Wh_plus_Ux = pmath.mobius_add(W_otimes_h, U_otimes_x, c=c)
    return pmath.mobius_add(Wh_plus_Ux, b, c=c)


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = pmath.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, c), c=c).sigmoid()
    r_t = pmath.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, c), c=c).sigmoid()

    rh_t = pmath.mobius_pointwise_mul(r_t, hx, c=c)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, c)

    if nonlin is not None:
        h_tilde = pmath.mobius_fn_apply(nonlin, h_tilde, c=c)
    delta_h = pmath.mobius_add(-hx, h_tilde, c=c)
    h_out = pmath.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, c=c), c=c)
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin=None,
):
    if not hyperbolic_hidden_state0:
        hx = pmath.expmap0(h0, c=c)
    else:
        hx = h0
    if not hyperbolic_input:
        input = pmath.expmap0(input, c=c)
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                c=c,
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                c=c,
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last


class MobiusLinear(torch.nn.Linear):
    def __init__(
        self,
        *args,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        c=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=c)
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() / 4, c=c))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            c=self.ball.c,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = geoopt.PoincareBall(c=c)
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 4
        point = pmath.expmap0(point, c=c)
        tangent = torch.randn(out_features, in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

    def forward(self, input):
        input = input.unsqueeze(-2)
        distance = pmath.dist2plane(
            x=input, p=self.point, a=self.tangent, c=self.ball.c, signed=True
        )
        return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}, "
            "c={ball.c}".format(
                **self.__dict__
            )
        )


class MobiusGRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlin=None,
        hyperbolic_input=True,
        hyperbolic_hidden_state0=True,
        c=1.0,
    ):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=c)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.weight_ih = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_hh = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.randn(3, hidden_size) * 1e-5
                bias = geoopt.ManifoldParameter(
                    pmath.expmap0(bias, c=self.ball.c), manifold=self.ball
                )
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)
        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
            )
        h0 = h0.unbind(0)
        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers
        outputs = []
        last_states = []
        out = input
        for i in range(self.num_layers):
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=biases[i],
                c=self.ball.c,
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                nonlin=self.nonlin,
                batch_sizes=batch_sizes,
            )
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)
        # default api assumes
        # out: (seq_len, batch, num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        # if packed:
        # out: (sum(seq_len), num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        return out, ht

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "c={self.ball.c}"
        ).format(**self.__dict__, self=self, bias=self.bias is not None)


class RNNBase(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        project_dim,
        cell_type="_rnn",
        embedding_type="eucl",
        decision_type="eucl",
        use_distance_as_feature=True,
        device=None,
        num_layers=1,
        num_classes=1,
        c=1.0,
    ):
        super(RNNBase, self).__init__()
        (cell_type, embedding_type, decision_type) = map(
            str.lower, [cell_type, embedding_type, decision_type]
        )
        if embedding_type == "eucl":
            self.embedding = LookupEmbedding(
                vocab_size, embedding_dim, manifold=geoopt.Euclidean()
            )
            with torch.no_grad():
                self.embedding.weight.normal_()
        elif embedding_type == "hyp":
            self.embedding = LookupEmbedding(
                vocab_size,
                embedding_dim,
                manifold=geoopt.PoincareBall(c=c),
            )
            with torch.no_grad():
                self.embedding.weight.set_(
                    pmath.expmap0(self.embedding.weight.normal_() / 10, c=c)
                )
        else:
            raise NotImplementedError(
                "Unsuported embedding type: {0}".format(embedding_type)
            )
        self.embedding_type = embedding_type
        if decision_type == "eucl":
            self.projector = nn.Linear(hidden_dim * 2, project_dim)
            self.logits = nn.Linear(project_dim, num_classes)
        elif decision_type == "hyp":
            self.projector_source = MobiusLinear(
                hidden_dim, project_dim, c=c
            )
            self.projector_target = MobiusLinear(
                hidden_dim, project_dim, c=c
            )
            self.logits = MobiusDist2Hyperplane(project_dim, num_classes)
        else:
            raise NotImplementedError(
                "Unsuported decision type: {0}".format(decision_type)
            )
        self.ball = geoopt.PoincareBall(c)
        if use_distance_as_feature:
            if decision_type == "eucl":
                self.dist_bias = nn.Parameter(torch.zeros(project_dim))
            else:
                self.dist_bias = geoopt.ManifoldParameter(
                    torch.zeros(project_dim), manifold=self.ball
                )
        else:
            self.register_buffer("dist_bias", None)
        self.decision_type = decision_type
        self.use_distance_as_feature = use_distance_as_feature
        self.device = device  # declaring device here due to fact we are using catalyst
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.c = c

        if cell_type == "eucl_rnn":
            self.cell = nn.RNN
        elif cell_type == "eucl_gru":
            self.cell = nn.GRU
        elif cell_type == "hyp_gru":
            self.cell = functools.partial(MobiusGRU, c=c)
        else:
            raise NotImplementedError("Unsuported cell type: {0}".format(cell_type))
        self.cell_type = cell_type

        self.cell_source = self.cell(embedding_dim, self.hidden_dim, self.num_layers)
        self.cell_target = self.cell(embedding_dim, self.hidden_dim, self.num_layers)

    def forward(self, input, _len):


        source_input = input #input[0]
        # target_input = input[1]
        # alignment = input[2]
        batch_size = source_input.shape[0]

        source_input_data = self.embedding(source_input.data)
        # target_input_data = self.embedding(target_input.data)

        zero_hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=self.device or source_input.device,
            dtype=source_input_data.dtype
        )

        if self.embedding_type == "eucl" and "hyp" in self.cell_type:
            source_input_data = pmath.expmap0(source_input_data, c=self.c)
            # target_input_data = pmath.expmap0(target_input_data, c=self.c)
        elif self.embedding_type == "hyp" and "eucl" in self.cell_type:
            source_input_data = pmath.logmap0(source_input_data, c=self.c)
            # target_input_data = pmath.logmap0(target_input_data, c=self.c)
        # ht: (num_layers * num_directions, batch, hidden_size)

        source_input = torch.nn.utils.rnn.PackedSequence(
            source_input_data, _len=_len, #source_input.batch_sizes
        )
        # target_input = torch.nn.utils.rnn.PackedSequence(
        #     target_input_data, target_input.batch_sizes
        # )

        _, source_hidden = self.cell_source(source_input, zero_hidden)
        # _, target_hidden = self.cell_target(target_input, zero_hidden)

        # take hiddens from the last layer
        source_hidden = source_hidden[-1]
        # target_hidden = target_hidden[-1][alignment]

        # if self.decision_type == "hyp":
        #     if "eucl" in self.cell_type:
        #         source_hidden = pmath.expmap0(source_hidden, c=self.c)
        #         # target_hidden = pmath.expmap0(target_hidden, c=self.c)
        #     source_projected = self.projector_source(source_hidden)
        #     # target_projected = self.projector_target(target_hidden)
        #     # projected = pmath.mobius_add(
        #     #     source_projected, target_projected, c=self.ball.c
        #     # )
        #     # if self.use_distance_as_feature:
        #     #     dist = (
        #     #         pmath.dist(source_hidden, target_hidden, dim=-1, keepdim=True, c=self.ball.c) ** 2
        #     #     )
        #     #     bias = pmath.mobius_scalar_mul(dist, self.dist_bias, c=self.ball.c)
        #     projected = pmath.mobius_add(source_projected, bias, c=self.ball.c)
        # else:
        #     if "hyp" in self.cell_type:
        #         source_hidden = pmath.logmap0(source_hidden, c=self.c)
        #         target_hidden = pmath.logmap0(target_hidden, c=self.c)
        #     projected = self.projector(
        #         torch.cat((source_hidden, target_hidden), dim=-1)
        #     )
        #     if self.use_distance_as_feature:
        #         dist = torch.sum(
        #             (source_hidden - target_hidden).pow(2), dim=-1, keepdim=True
        #         )
        #         bias = self.dist_bias * dist
        #         projected = projected + bias

        logits = self.logits(source_hidden)
        # CrossEntropy accepts logits
        return logits
