from os import cpu_count

import mimikit as mmk
import h5mapper as h5m


class Seq2SeqLSTM(mmk.Seq2SeqLSTM):
    feature = mmk.Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')

    def __init__(self,
                 feature=None,
                 input_heads=1,
                 output_heads=4,
                 scaled_activation=True,
                 with_tbptt=False,
                 **net_hp):
        model_dim = net_hp['model_dim']
        output_mod = feature.output_module(model_dim, feature.n_fft // 2 + 1, output_heads, scaled_activation)
        net_hp["input_dim"] = feature.n_fft // 2 + 1
        net_hp["input_module"] = feature.input_module(net_hp["input_dim"], model_dim, input_heads)
        net_hp["output_module"] = output_mod
        super(Seq2SeqLSTM, self).__init__(**net_hp)
        self.hp.feature = self.feature = feature
        self.hp.input_heads = input_heads
        self.hp.output_heads = output_heads
        self.hp.scaled_activation = scaled_activation
        self.hp.with_tbptt = with_tbptt

    def train_dataloader(self, soundbank, batch_size, batch_length,
                         downsampling=1, shift_error=0, batch_sampler=None):
        hop = self.hp.hop
        feat = self.feature
        batch = (
            self.feature.batch_item(shift=0, length=batch_length, downsampling=downsampling),
            self.feature.batch_item(shift=hop+shift_error, length=batch_length, downsampling=downsampling),
        )
        if self.hp.with_tbptt:
            kwargs = dict(batch_sampler=mmk.TBPTTSampler(soundbank.snd.shape[0] // feat.hop_length,
                                                         batch_size=batch_size,
                                                         chunk_length=batch_length * 20,
                                                         seq_len=batch_length))
        else:
            kwargs = dict(batch_size=batch_size, shuffle=True)

        dl = soundbank.serve(batch,
                             num_workers=min(batch_size, cpu_count()),
                             pin_memory=True,
                             persistent_workers=True,
                             **kwargs)
        return dl

    def generate_dataloader_and_interfaces(self,
                                           soundbank,
                                           prompt_length,
                                           indices=(),
                                           batch_size=256,
                                           temperature=None,
                                           **kwargs
                                           ):
        gen_batch = (self.feature.batch_item(shift=0, length=prompt_length),)
        gen_dl = soundbank.serve(gen_batch,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 sampler=indices)

        interfaces = [
            *(mmk.DynamicDataInterface(
                None,
                getter=h5m.AsSlice(dim=1, shift=-self.hp.hop, length=self.hp.hop),
                setter=mmk.Setter(dim=1),
                # output_transform=(lambda x: x.unsqueeze(1) if isinstance(self.feature, mmk.Spectrogram) else x)
            ),),
        ]

        return gen_dl, interfaces
