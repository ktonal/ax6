from functools import partial
import h5mapper as h5m
import mimikit as mmk
import torch


class Seq2SeqLSTMv0(mmk.Seq2SeqLSTM):
    feature = mmk.Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')
    """no input module"""
    def __init__(self,
                 feature=None,
                 output_heads=4,
                 scaled_activation=True,
                 **net_hp):
        model_dim = net_hp['model_dim']
        is_pol = (1+int(feature.coordinate == 'pol'))
        fft_dim = (feature.n_fft // 2 + 1) * is_pol
        output_mod = feature.output_module(model_dim, fft_dim//is_pol, output_heads, scaled_activation)
        net_hp["input_dim"] = fft_dim
        net_hp["output_module"] = output_mod
        if feature.coordinate == "pol":
            net_hp["input_module"] = mmk.Flatten(2)
        super(Seq2SeqLSTMv0, self).__init__(**net_hp)
        self.hp.feature = self.feature = feature
        self.hp.output_heads = output_heads
        self.hp.scaled_activation = scaled_activation


class Seq2SeqLSTM(mmk.Seq2SeqLSTM):
    feature = mmk.Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')

    def __init__(self,
                 feature=None,
                 input_heads=1,
                 output_heads=4,
                 scaled_activation=True,
                 **net_hp):
        model_dim = net_hp['model_dim']
        output_mod = feature.output_module(model_dim, feature.n_fft // 2 + 1, output_heads, scaled_activation)
        net_hp["input_dim"] = model_dim
        net_hp["input_module"] = feature.input_module(feature.n_fft // 2 + 1, model_dim, input_heads)
        net_hp["output_module"] = output_mod
        super(Seq2SeqLSTM, self).__init__(**net_hp)
        self.hp.feature = self.feature = feature
        self.hp.input_heads = input_heads
        self.hp.output_heads = output_heads
        self.hp.scaled_activation = scaled_activation

        def frw_hook(module, inpt, output):
            is_tensor = lambda x: isinstance(x, torch.Tensor)
            h5m.process_batch(inpt, is_tensor,
                              lambda x: print("INPUT NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            h5m.process_batch(output, is_tensor,
                              lambda x: print("OUTPUT NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            return output

        def grad_hook(module, inpt, output):
            is_tensor = lambda x: isinstance(x, torch.Tensor)
            h5m.process_batch(inpt, is_tensor,
                              lambda x: print("INPUT GRAD NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            h5m.process_batch(output, is_tensor,
                              lambda x: print("OUTPUT GRAD NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            return inpt

        # for mod in self.modules():
        #     mod.register_full_backward_hook(grad_hook)
            # mod.register_forward_hook(frw_hook)

        # def hook(grad):
        #     is_nans = torch.isnan(grad)
        #     if torch.any(is_nans):
        #         return torch.where(is_nans, torch.zeros_like(grad), grad)
        #     return grad
        # for p in self.parameters():
        #     p.register_hook(hook)


class Seq2SeqMuLaw(mmk.Seq2SeqLSTM):
    feature = mmk.MuLawSignal(sr=16000, q_levels=256, normalize=True)

    def __init__(self, feature=None, mlp_dim=128, **net_hp):
        net_hp["input_module"] = feature.input_module(net_hp["input_dim"])
        out_d = net_hp["model_dim"]
        net_hp["output_module"] = feature.output_module(out_d, mlp_dim=mlp_dim)
        super(Seq2SeqMuLaw, self).__init__(**net_hp)
        self.hp.feature = self.feature = feature
        self.hp.mlp_dim = mlp_dim


class MultiSeq2SeqFFT(mmk.MultiSeq2SeqLSTM):
    pass


class MultiSeq2SeqMuLaw(mmk.MultiSeq2SeqLSTM):
    pass
