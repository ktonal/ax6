import dataclasses as dtc
from typing import List, Any, Dict

import h5mapper as h5m
import mimikit as mmk
import json
import os, sys

dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path:
    sys.path.append(dir1)

from ax6.datasets import TRAINSET, gcp_sound_bank


def find_checkpoints(root="trainings"):
    return h5m.FileWalker(r"\.h5", root)


def load_trainings_hp(dirname):
    return json.loads(open(os.path.join(dirname, "hp.json"), 'r').read())


class CkptBank(h5m.TypedFile):
    ckpt = h5m.TensorDict()


def load_feature(s):
    import mimikit as mmk
    loc = dict()
    exec(f"feature = {s}", mmk.__dict__, loc)
    return loc["feature"]


def load_network_cls(s):
    from ax6.models.s2s import Seq2SeqLSTM
    from ax6.models.wavenets import WaveNetFFT, WaveNetQx
    from ax6.models.srnns import SampleRNN
    loc = dict()
    exec(f"cls = {s}", locals(), loc)
    return loc["cls"]


def load_files(files, sr):
    if "gs://" in files[0]:
        sb = gcp_sound_bank(sr)
        sb.create(f"data.h5", files, parallelism="threads", n_workers=8)
        return sb(f"data.h5", mode='r', keep_open=True)
    else:
        h5m.sound_bank.callback(f"data.h5", files[0], sr=sr)
        return h5m.TypedFile("data.h5", mode='r', keep_open=True)


def match_trainset_id(files, dirname, hp):
    if "trainset" in hp:
        return [c for c in TRAINSET if c["keywords"] == hp["trainset"]][0]
    if "verdi-x" in dirname:
        kwrds = f"Verdi_X_{os.path.split(files[0])[-1].split('_')[2]}"

    else:
        paths = {"blobs/" + os.path.split(f)[-1] for f in files}
        for col in TRAINSET:
            blobs_set = {b["path"] for b in col["blobs"]}
            if paths == blobs_set or (
                    paths < blobs_set and ("insects-" in (*paths,)[0] or "dies-irae" in (*paths,)[0])):
                return col
        return {"***************************FAILED*******", tuple(files)}

    coll = [col for col in TRAINSET if kwrds in col["keywords"]][0]
    return coll


@dtc.dataclass
class Checkpoint:
    net_cls: type
    ckpt_bank: h5m.TypedFile
    feature: mmk.Feature
    epochs: List[str]
    hp: Dict[str, Any]


def group_ckpts_by_trainset(root="trainings"):
    CKPTS = {}
    for ckpt in find_checkpoints(root):
        tp = CkptBank(ckpt, "r")
        dirname = os.path.dirname(ckpt)
        hp = load_trainings_hp(dirname)
        feature = load_feature(hp["network"]["feature"])
        trainset_coll = match_trainset_id(hp["files"], dirname, hp)
        # todo : cache trainset id in hp.json

        #     files = load_files(hp["files"], feature.sr)
        try:
            src = [*tp.index.keys()][0]
        except IndexError:
            raise IndexError(dirname)
        try:
            tp.ckpt.load_checkpoint(load_network_cls(hp["network_class"]), src)
            CKPTS.setdefault(trainset_coll["id"], []).append(
                (load_network_cls(hp["network_class"]), tp, feature, [*tp.index.keys()], hp)
            )
            # print("+++OK++++", dirname)

        except RuntimeError:  # miss matched state dict!
            # print(type(e))
            pass
            # print("---NO!----", dirname)
    return CKPTS


def group_ckpts_by_feature(root="trainings"):
    CKPTS = {}
    for ckpt in find_checkpoints(root):
        tp = CkptBank(ckpt, "r")
        dirname = os.path.dirname(ckpt)
        hp = load_trainings_hp(dirname)
        feature = load_feature(hp["network"]["feature"])
        CKPTS.setdefault(feature, []).append(
            (load_network_cls(hp["network_class"]), tp, feature, [*tp.index.keys()], hp)
        )
    return CKPTS
