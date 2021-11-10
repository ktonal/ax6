import mimikit as mmk
from itertools import chain


class WaveNetQx(mmk.WNBlock):
    feature = mmk.MuLawSignal(sr=16000, q_levels=256, normalize=True)

    def __init__(self, feature=None, mlp_dim=128, **block_hp):
        super(WaveNetQx, self).__init__(**block_hp)
        if feature is not None:
            self.hp.feature = self.feature = feature
        self.hp.mlp_dim = mlp_dim
        inpt_mods = [feat.input_module(d)
                     for feat, d in zip([feature], chain(self.hp.dims_dilated, self.hp.dims_1x1))]
        out_d = self.hp.skips_dim if self.hp.skips_dim is not None else self.hp.dims_dilated[0]
        outpt_mods = [feat.output_module(out_d, mlp_dim=mlp_dim)
                      for feat in [feature]]
        self.with_io(inpt_mods, outpt_mods)


class WaveNetFFT(mmk.WNBlock):
    feature = mmk.Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')

    def __init__(self,
                 feature,
                 input_heads=2,
                 output_heads=4,
                 scaled_activation=True,
                 **block_hp
                 ):
        super(WaveNetFFT, self).__init__(**block_hp)
        if feature is not None:
            self.hp.feature = self.feature = feature
        self.hp.input_heads = input_heads
        self.hp.output_heads = output_heads
        self.hp.scaled_activation = scaled_activation
        fft_dim = self.feature.n_fft // 2 + 1
        net_dim = self.hp.dims_dilated[0]
        inpt_mods = [self.feature.input_module(fft_dim, net_dim, input_heads)]
        out_d = self.hp.skips_dim if self.hp.skips_dim is not None else self.hp.dims_dilated[0]
        outpt_mods = [self.feature.output_module(out_d, fft_dim, output_heads,
                                                 scaled_activation=scaled_activation)]
        self.with_io(inpt_mods, outpt_mods)
