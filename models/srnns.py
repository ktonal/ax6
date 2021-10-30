from os import cpu_count

import mimikit as mmk
import h5mapper as h5m


class SRNNBase(mmk.TierNetwork):
    feature: mmk.Feature = None

    def train_dataloader(self, soundbank, batch_size, batch_length,
                         downsampling=None, shift_error=0, **kwargs):
        batch = (tuple(
            self.feature.batch_item(shift=self.shift - fs,
                                    length=batch_length, frame_size=fs,
                                    as_strided=False)
            for fs in self.frame_sizes[:-1]
        ) + (
                     self.feature.batch_item(shift=self.shift - self.frame_sizes[-1],
                                             length=batch_length, frame_size=self.frame_sizes[-1],
                                             as_strided=True),
                 ),
                 self.feature.batch_item(shift=self.shift + shift_error,
                                         length=batch_length,)
                 )
        dl = soundbank.serve(batch,
                             num_workers=min(batch_size, cpu_count()),
                             pin_memory=True,
                             persistent_workers=True,  # need this!
                             batch_sampler=mmk.TBPTTSampler(soundbank.snd.shape[0],
                                                            batch_size=batch_size,
                                                            chunk_length=self.hp.chunk_length,
                                                            seq_len=batch_length,
                                                            downsampling=downsampling))
        return dl

    def generate_dataloader_and_interfaces(self,
                                           soundbank,
                                           prompt_length,
                                           indices=(),
                                           batch_size=256,
                                           temperature=None,
                                           **kwargs
                                           ):
        gen_batch = (self.feature.batch_item(shift=0, length=prompt_length, ),)
        gen_dl = soundbank.serve(gen_batch,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 sampler=indices
                                 )
        interfaces = [
            mmk.DynamicDataInterface(
                None,
                getter=h5m.AsSlice(dim=1, shift=-self.shift, length=self.shift),
                setter=mmk.Setter(dim=1)
            ),
            # temperature
            *((mmk.DynamicDataInterface(
                None if callable(temperature) else temperature,
                prepare=(lambda src: temperature()) if callable(temperature) else lambda src: src,
                getter=h5m.AsSlice(dim=1, shift=0, length=1),
                setter=None,
            ),) if temperature is not None else ())
        ]
        return gen_dl, interfaces


class SampleRNN(mmk.SampleRNN, SRNNBase):

    def __init__(self, feature, chunk_length=16000 * 8, **net_hp):
        super(SampleRNN, self).__init__(**net_hp)
        self.feature = self.hp.feature = feature
        self.hp.chunk_length = chunk_length
