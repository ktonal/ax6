import json
import os
import hashlib
import mimikit as mmk
import h5mapper as h5m
import torch


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super(MyEncoder, self).default(obj)
        except:
            return str(obj)


def train(
        soundbank,
        net,

        batch_size=16,
        batch_length=32,
        downsampling=1,
        shift_error=0,

        max_epochs=2,
        limit_train_batches=1000,
        max_lr=5e-4,
        betas=(0.9, 0.93),
        div_factor=3.,
        final_div_factor=1.,
        pct_start=0.,
        cycle_momentum=False,

        CHECKPOINT_TRAINING=True,

        MONITOR_TRAINING=True,
        OUTPUT_TRAINING='',

        every_n_epochs=2,
        n_examples=3,
        prompt_length=32,
        n_steps=200,
        temperature=torch.tensor([[.85] * 200]),
):
    train_hp = dict(locals())
    train_hp.pop("net")
    train_hp.pop("soundbank")
    train_hp.pop("temperature")
    hp = dict(files=list(soundbank.index.keys()),
              network_class=net.__class__.__qualname__,
              network=net.hp,
              train_hp=train_hp)
    ID = hashlib.sha256(json.dumps(hp, cls=MyEncoder).encode("utf-8")).hexdigest()
    print("****************************************************")
    print("ID IS :", ID)
    print("****************************************************")
    hp['id'] = ID
    os.makedirs(f"trainings/{ID}/outputs", exist_ok=True)
    filename_template = f"trainings/{ID}/outputs/" + "epoch{epoch}_prm{prompt_idx}.wav"
    with open(f"trainings/{ID}/hp.json", "w") as fp:
        json.dump(hp, fp, cls=MyEncoder)
    logs_file = f"trainings/{ID}/checkpoints.h5"

    dl = net.train_dataloader(soundbank,
                              batch_size=batch_size,
                              batch_length=batch_length,
                              downsampling=downsampling,
                              shift_error=shift_error,
                              )

    opt = torch.optim.Adam(net.parameters(), lr=max_lr, betas=betas)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        steps_per_epoch=min(len(dl), limit_train_batches) if limit_train_batches is not None else len(dl),
        epochs=max_epochs,
        max_lr=max_lr,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        pct_start=pct_start,
        cycle_momentum=cycle_momentum
    )

    tr_loop = mmk.TrainLoop(
        loader=dl,
        net=net,
        loss_fn=net.feature.loss_fn,
        optim=([opt], [{"scheduler": sched, "interval": "step", "frequency": 1}])
    )

    # Gen Loop
    max_i = soundbank.snd.shape[0] - getattr(net.feature, "hop_length", 1) * prompt_length
    g_dl, g_interfaces = net.generate_dataloader_and_interfaces(
        soundbank,
        prompt_length=prompt_length,
        indices=mmk.IndicesSampler(N=n_examples,
                                   max_i=max_i,
                                   redraw=True),
        temperature=temperature
    )
    gen_loop = mmk.GenerateLoop(
        network=net,
        dataloader=g_dl,
        interfaces=g_interfaces,
        n_steps=n_steps,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    class Logs(h5m.TypedFile):
        ckpt = h5m.TensorDict(net.state_dict()) if CHECKPOINT_TRAINING else None
        outputs = h5m.Array() if 'h5' in OUTPUT_TRAINING else None

    logs = Logs(logs_file, mode='w')
    callbacks = []

    if CHECKPOINT_TRAINING:
        callbacks += [
            mmk.MMKCheckpoint(h5_tensor_dict=logs.ckpt, epochs=every_n_epochs)
        ]

    if MONITOR_TRAINING or OUTPUT_TRAINING:
        callbacks += [
            mmk.GenerateCallback(
                generate_loop=gen_loop,
                every_n_epochs=every_n_epochs,
                output_features=[net.feature, ],
                audio_logger=mmk.AudioLogger(
                    sr=net.feature.sr,
                    hop_length=getattr(net.feature, 'hop_length', 512),
                    **(dict(filename_template=filename_template,
                            target_dir=os.path.dirname(filename_template))
                       if 'mp3' in OUTPUT_TRAINING else {}),
                    **(dict(id_template="idx_{prompt_idx}",
                            proxy_template="outputs/epoch_{epoch}/",
                            target_bank=logs)
                       if 'h5' in OUTPUT_TRAINING else {})
                ),
            )]

    gen_loop.plot_audios = gen_loop.play_audios = MONITOR_TRAINING

    tr_loop.run(max_epochs=max_epochs,
                logger=None,
                callbacks=callbacks,
                limit_train_batches=limit_train_batches if limit_train_batches is not None else 1.
                )

    soundbank.close()
    os.remove("train.h5")


def generate(
        soundbank,
        net,
        filename_template=None,
        target_dir=None,
        id_template=None,
        proxy_template=None,
        target_bank=None,
        indices=(),
        prompt_length=32,
        n_steps=200,
        temperature=None,
):
    g_dl, g_interfaces = net.generate_dataloader_and_interfaces(
        soundbank,
        prompt_length=prompt_length,
        indices=indices,
        temperature=temperature
    )
    mp3 = filename_template and target_dir
    h5 = id_template and proxy_template and target_bank

    logger = mmk.AudioLogger(
        sr=net.feature.sr,
        hop_length=getattr(net.feature, 'hop_length', 512),
        **(dict(filename_template=filename_template,
                target_dir=target_dir)
           if mp3 else {}),
        **(dict(id_template=id_template,
                proxy_template=proxy_template,
                target_bank=target_bank)
           if h5 else {})
    )

    def process_outputs(outputs, batch_idx):
        output = net.feature.inverse_transform(outputs[0])
        for i, out in enumerate(output):
            idx = batch_idx * g_dl.batch_size + i
            prompt_idx = indices[idx]
            logger.write(out, prompt_idx=prompt_idx)
        return

    gen_loop = mmk.GenerateLoop(
        network=net,
        dataloader=g_dl,
        interfaces=g_interfaces,
        n_steps=n_steps,
        process_outputs=process_outputs,
        # time_hop=getattr(net.feature, 'hop_length', 1),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    gen_loop.run()
