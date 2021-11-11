import mimikit as mmk


class Seq2SeqLSTMv0(mmk.Seq2SeqLSTM):
    feature = mmk.Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')
    """no input module"""
    def __init__(self,
                 feature=None,
                 output_heads=4,
                 scaled_activation=True,
                 with_tbptt=False,
                 **net_hp):
        model_dim = net_hp['model_dim']
        output_mod = feature.output_module(model_dim, feature.n_fft // 2 + 1, output_heads, scaled_activation)
        net_hp["input_dim"] = feature.n_fft // 2 + 1
        net_hp["output_module"] = output_mod
        super(Seq2SeqLSTMv0, self).__init__(**net_hp)
        self.hp.feature = self.feature = feature
        self.hp.output_heads = output_heads
        self.hp.scaled_activation = scaled_activation
        self.hp.with_tbptt = with_tbptt


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


class Seq2SeqMuLaw(mmk.Seq2SeqLSTM):
    pass


class MultiSeq2SeqFFT(mmk.MultiSeq2SeqLSTM):
    pass


class MultiSeq2SeqMuLaw(mmk.MultiSeq2SeqLSTM):
    pass
