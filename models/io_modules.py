import mimikit as mmk
import torch.nn as nn
import torch


def qx_io(q_levels, net_dim, mlp_dim, mlp_activation=nn.ReLU()):
    return nn.Embedding(q_levels, net_dim), \
           mmk.SingleClassMLP(net_dim, mlp_dim, q_levels, mlp_activation)


def mag_spec_io(spec_dim, net_dim, in_chunks, out_chunks, scaled_activation=False):
    return mmk.Chunk(nn.Linear(spec_dim, net_dim * in_chunks),
                     in_chunks, sum_out=True), \
           nn.Sequential(mmk.Chunk(nn.Linear(net_dim, spec_dim * out_chunks),
                                   out_chunks, sum_out=True),
                         mmk.ScaledSigmoid(spec_dim, with_range=False) if scaled_activation else mmk.Abs())


def pol_spec_io(spec_dim, net_dim, in_chunks, out_chunks, scaled_activation=False):
    pi = torch.acos(torch.zeros(1)).item()
    act_phs = mmk.ScaledTanh(spec_dim, with_range=False) if scaled_activation else nn.Tanh()
    act_mag = mmk.ScaledSigmoid(spec_dim, with_range=False) if scaled_activation else mmk.Abs()

    class ScaledPhase(mmk.HOM):
        def __init__(self):
            super(ScaledPhase, self).__init__(
                "x -> phs",
                (nn.Sequential(mmk.Chunk(nn.Linear(net_dim, spec_dim * out_chunks), out_chunks, sum_out=True), act_phs),
                 'x -> phs'),
                (lambda self, phs: torch.cos(
                    phs * self.psis.to(phs).view(*([1] * (len(phs.shape) - 1)), -1)) * pi,
                 'self, phs -> phs'),
            )
            self.psis = nn.Parameter(torch.ones(spec_dim))

    return mmk.HOM('x -> x',
                   (mmk.Flatten(2), 'x -> x'),
                   (mmk.Chunk(
                       nn.Linear(spec_dim * 2, net_dim * in_chunks),
                       in_chunks, sum_out=True),
                    'x -> x')), \
           mmk.HOM("x -> y",
                   # phase module
                   (ScaledPhase(), 'x -> phs'),
                   # magnitude module
                   (nn.Sequential(mmk.Chunk(
                       nn.Linear(net_dim, spec_dim * out_chunks),
                       out_chunks, sum_out=True),
                       act_mag),
                    'x -> mag'),
                   (lambda mag, phs: torch.stack((mag, phs), dim=-1), "mag, phs -> y")
                   )
