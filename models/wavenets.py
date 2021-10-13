import mimikit as mmk
import h5mapper as h5m
from itertools import chain


class WaveNetBase(mmk.WNBlock):

    feature: mmk.Feature = None

    def train_dataloader(self, soundbank, batch_size, batch_length,
                         downsampling=1, shift_error=0, batch_sampler=None):
        getters = self.getters(batch_length=batch_length,
                               downsampling=downsampling,
                               hop_length=getattr(self.feature, 'hop_length', 1),
                               shift_error=shift_error
                               )
        batch = (
            h5m.Input(key='snd', getter=getters['inputs'], transform=self.feature.transform),
            h5m.Target(key='snd', getter=getters['targets'], transform=self.feature.transform),
        )
        return soundbank.serve(batch,
                               batch_size=batch_size,
                               num_workers=min(batch_size, 16),
                               pin_memory=True,
                               persistent_workers=True,  # need this!
                               shuffle=True,
                               batch_sampler=batch_sampler
                               )

    def generate_dataloader_and_interfaces(self,
                                           soundbank,
                                           prompt_length,
                                           indices=(),
                                           batch_size=256,
                                           temperature=None,
                                           **kwargs
                                           ):
        gen_getters = self.getters(batch_length=prompt_length,
                                   downsampling=1,
                                   hop_length=getattr(self.feature, 'hop_length', 1),
                                   shift_error=0)
        gen_batch = (h5m.Input(key="snd",
                               getter=gen_getters['inputs'],
                               transform=self.feature.transform),)
        gen_dl = soundbank.serve(gen_batch,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 sampler=indices)
        interfaces = [
            *(mmk.DynamicDataInterface(
                None,
                getter=h5m.AsSlice(dim=1, shift=-self.rf, length=self.rf),
                setter=mmk.Setter(dim=1),
                output_transform=(lambda x: x.unsqueeze(1) if isinstance(self.feature, mmk.Spectrogram) else x)
            ),),
            # temperature
            *((mmk.DynamicDataInterface(
                None if callable(temperature) else temperature,
                prepare=(lambda src: temperature()) if callable(temperature) else lambda src: src,
                getter=h5m.AsSlice(dim=1, shift=0, length=1),
                setter=None,
            ),) if temperature is not None and "temperature" in self.s else ())
        ]
        return gen_dl, interfaces


class WaveNetQx(WaveNetBase):

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


class WaveNetFFT(WaveNetBase):

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

