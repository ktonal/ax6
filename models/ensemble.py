import mimikit as mmk
import h5mapper as h5m
import torch.nn as nn
import torch
from pytorch_lightning.utilities import AttributeDict
import math


class Events(AttributeDict):
    pass


base_sr = 44100
feat = mmk.MuLawSignal(sr=16000, q_levels=256)


class Ensemble(nn.Module):
    device = "cuda"

    def __init__(self,
                 base_sr,
                 stream,
                 *net_features_pairs):
        super(Ensemble, self).__init__()
        networks, features = zip(*net_features_pairs)
        self.networks = nn.ModuleList(networks)
        self.features = features
        self.base_sr = base_sr

    def run_event(self, inputs, net, feature, n_steps):
        prompt_getter = feature.batch_item(shift=0, length=net.rf)
        resample = mmk.Resample(self.base_sr, feature.sr)
        n_input_samples = math.ceil(prompt_getter.getter.length * self.base_sr / feature.sr)
        prompt_getter.data = resample(inputs[:, -n_input_samples:])
        prompt = prompt_getter(0)
        output = [[]]

        def process_outputs(outputs, _):
            inv_resample = mmk.Resample(feature.sr, self.base_sr)
            # prompt + generated in base_sr :
            y = feature.inverse_transform(outputs[0])
            out = inv_resample(y)
            output[0] = out[:, n_input_samples:]

        loop = mmk.GenerateLoop(net,
                                dataloader=[(prompt,)],
                                inputs=(h5m.Input(None,
                                                  getter=h5m.AsSlice(dim=1, shift=-net.shift, length=net.shift),
                                                  setter=h5m.Setter(dim=1)),),
                                n_steps=n_steps,
                                add_blank=True,
                                process_outputs=process_outputs,
                                time_hop=getattr(net, 'hop', 1)
                                )
        loop.run()
        return output[0]

    @staticmethod
    def seconds_to_n_steps(seconds, net, feature):
        return int(seconds * feature.sr) if isinstance(feature, mmk.MuLawSignal) \
            else int(seconds*(feature.sr // feature.hop_length)) // getattr(net, "hop", 1)

    def generate_step(self, t, inputs, ctx):
        net, feature, seconds = next(self.stream)
        net = net.to("cuda")
        if hasattr(net, "use_fast_generate"):
            net.use_fast_generate = True
        n_steps = self.seconds_to_n_steps(seconds, net, feature)

        if (t / 16000) < (N_SECONDS - 1):
            out = self.run_event(inputs[0], net, feature, n_steps)
            return out
        return torch.zeros(1, (N_SECONDS + 1) * 16000 - t).to("cuda")
