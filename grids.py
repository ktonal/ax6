from sklearn.model_selection import ParameterGrid, ParameterSampler
import mimikit as mmk
from itertools import product
import numpy as np


def instance_grid(cls, fix, *grids):
    for params in product(*grids):
        d = {}
        for p in params:
            d.update(**p)
        if cls is not None:
            yield cls(**fix, **d)
        else:
            yield dict(**fix, **d)


def n_choices(iterable, n):
    lst = list(iterable)
    for x in np.random.choice(lst, n, replace=len(lst) < n):
        yield x


def sampler_zipper(n_samples, *grids):
    for gs in zip(*(n_choices(g, n_samples) for g in grids)):
        yield {k: v for g in gs for k, v in g.items()}


def fft_grid():
    return ParameterGrid(dict(feature=instance_grid(
        mmk.Spectrogram, dict(center=False),
        ParameterGrid([
            dict(sr=[44100],
                 )]),
        [
            # dict(n_fft=512, hop_length=128),
            # dict(n_fft=1024, hop_length=128),
            # dict(n_fft=1024, hop_length=256),
            dict(n_fft=2048, hop_length=256),
            # dict(n_fft=2048, hop_length=512),
        ],
        ParameterGrid([
            dict(coordinate=["mag"])
        ]),
    )))


def fft_io_grid():
    return instance_grid(
        None, {},
        ParameterGrid([
            dict(
                input_heads=[1],
                output_heads=[1],
                scaled_activation=[True, False])  # True seems ok but WITHOUT SAMPLER
        ])
    )


def optim_grid():
    return instance_grid(
        None,
        dict(),
        ParameterGrid([
            dict(
                batch_size=[8, 16, 24, 32],
                max_lr=[7e-4, 5e-4, 2e-4],
                betas=[(0.9, 0.91), (0.9, 0.92), (0.9, 0.925), (0.9, 0.95), (0.99, 0.99)],
            )
        ])
    )


WN_GRID = ParameterGrid([
    dict(
        # WN 44k
        network=dict(
            feature=mmk.Spectrogram(sr=44100,
                                    n_fft=1024,
                                    hop_length=128,
                                    coordinate='mag',
                                    center=False,
                                    normalize=True),
            input_heads=2,
            output_heads=2,
            scaled_activation=False,

            kernel_sizes=(2,),
            blocks=(3,),
            dims_dilated=(512,),
            dims_1x1=(),
            residuals_dim=None,
            apply_residuals=False,
            skips_dim=None,
            groups=2,
            pad_side=0,
            stride=1,
            bias=True,
        ),
        train=dict(
            batch_size=8,
            batch_length=32,
            downsampling=64,
            shift_error=0,
            max_epochs=25,
            max_lr=5e-4,
            betas=(0.9, 0.92),
            div_factor=5.,
            final_div_factor=1.,
            pct_start=0.,
        )),
    dict(
        # WN Qx
        network=dict(
            feature=mmk.MuLawSignal(sr=16000,
                                    q_levels=128),
            mlp_dim=512,

            kernel_sizes=(4,),
            blocks=(5,),
            dims_dilated=(512,),
            dims_1x1=(),
            residuals_dim=None,
            apply_residuals=False,
            skips_dim=None,
            groups=4,
            pad_side=0,
            stride=1,
            bias=True,
        ),
        train=dict(
            batch_size=4,
            batch_length=2048,
            downsampling=1,
            shift_error=0,
            max_epochs=50,
            max_lr=5e-4,
            betas=(0.999, 0.999),
            div_factor=5.,
            final_div_factor=1.,
            pct_start=0.,
            cycle_momentum=False,
        )),
])

S2S_GRID = ParameterGrid([
    dict(
        # S2S 44k
        network=dict(
            feature=mmk.Spectrogram(sr=44100,
                                    n_fft=1024, hop_length=256,
                                    coordinate='mag',
                                    center=False,
                                    normalize=True),
            input_heads=1,
            output_heads=1,
            scaled_activation=False,
            model_dim=256,
            num_layers=1,
            n_lstm=1,
            bottleneck="add",
            n_fc=1,
            hop=4,
            weight_norm=False,
            with_tbptt=False,
        ),
        train=dict(
            batch_size=8,
            batch_length=4,
            downsampling=32,
            shift_error=0,
            max_epochs=50,
            max_lr=4e-4,
            betas=(0.9, 0.92),
            div_factor=5.,
            final_div_factor=1.,
            pct_start=0.,
            cycle_momentum=False,
        )),
    dict(
        # S2S 22.5K
        network=dict(
            feature=mmk.Spectrogram(sr=22050,
                                    n_fft=2048, hop_length=512,
                                    coordinate='mag',
                                    center=False,
                                    normalize=True),
            input_heads=1,
            output_heads=2,
            scaled_activation=False,
            model_dim=512,
            num_layers=1,
            n_lstm=1,
            bottleneck="add",
            n_fc=1,
            hop=4,
            weight_norm=False,
            with_tbptt=False,
        ),
        train=dict(
            batch_size=8,
            batch_length=4,
            downsampling=64,
            shift_error=0,
            max_epochs=50,
            max_lr=4e-4,
            betas=(0.9, 0.9),
            div_factor=5.,
            final_div_factor=1.,
            pct_start=0.,
            cycle_momentum=False,
        )),
])
