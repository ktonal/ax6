import mimikit as mmk
import h5mapper as h5m


class SRNNBase(mmk.TierNetwork):

    feature: mmk.Feature = None

    def train_dataloader(self, soundbank, batch_size, batch_length,
                         downsampling=1, shift_error=0,
                         chunk_length=16000 * 8):
        getters = self.getters(batch_length=batch_length, shift_error=shift_error)
        batch = (
            tuple(h5m.Input(key='snd', getter=g_input,
                            transform=self.feature.transform)
                  for g_input in getters['inputs']),
            h5m.Target(key='snd', getter=getters['targets'],
                       transform=self.feature.transform),
        )
        dl = soundbank.serve(batch,
                             num_workers=min(batch_size, 16),
                             pin_memory=True,
                             persistent_workers=True,  # need this!
                             batch_sampler=mmk.TBPTTSampler(soundbank.snd.shape[0],
                                                            batch_size=batch_size,
                                                            chunk_length=chunk_length,
                                                            seq_len=batch_length))
        return dl

    def generate_dataloader_and_interfaces(self,
                                           soundbank,
                                           prompt_length,
                                           indices=(),
                                           batch_size=256,
                                           temperature=None,
                                           **kwargs
                                           ):
        gen_batch = (h5m.Input(key='snd',
                               getter=h5m.AsSlice(dim=0, shift=0, length=prompt_length),
                               transform=self.feature.transform),)
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
