import mimikit as mmk
import h5mapper as h5m
import torch.nn as nn
import torch
import math
from pprint import pprint
import os, sys

dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)

from ax6.checkpoints import Checkpoint


class Ensemble(nn.Module):
    device = "cuda"

    def __init__(self,
                 max_seconds,
                 base_sr,
                 stream,
                 print_events=False
                 ):
        super(Ensemble, self).__init__()
        self.max_seconds = max_seconds
        self.base_sr = base_sr
        self.stream = stream
        self.print_events = print_events

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
        if t >= int(self.max_seconds * self.base_sr):
            return None
        event, net, feature, n_steps = self.next_event()
        net = net.to("cuda")
        if hasattr(net, "use_fast_generate"):
            net.use_fast_generate = True

        if (t/self.base_sr+event['seconds']) < self.max_seconds:
            if self.print_events:
                event.update({"start": t/self.base_sr})
                pprint(event, sort_dicts=False, compact=True)
            out = self.run_event(inputs[0], net, feature, n_steps)
            return out
        return torch.zeros(1, int(self.max_seconds * self.base_sr - t)).to("cuda")

    def next_event(self):
        event = next(self.stream)
        ck = Checkpoint(event['id'], event['epoch'])
        net, feature = ck.network, ck.feature
        n_steps = self.seconds_to_n_steps(event['seconds'], net, feature)
        return event, net, feature, n_steps
