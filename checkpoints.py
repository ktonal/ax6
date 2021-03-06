import dataclasses as dtc
from google.cloud import storage
import h5mapper as h5m
import json
import os, sys

dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if dir1 not in sys.path:
    sys.path.append(dir1)

from ax6.datasets import TRAINSET


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
    from ax6.models.s2s import Seq2SeqLSTM, Seq2SeqLSTMv0
    from ax6.models.wavenets import WaveNetFFT, WaveNetQx
    from ax6.models.srnns import SampleRNN
    loc = dict()
    exec(f"cls = {s}", locals(), loc)
    return loc["cls"]


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


def group_ckpts_by_trainset(root="trainings", assert_loadable=False):
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
        if assert_loadable:
            tp.ckpt.load_checkpoint(load_network_cls(hp["network_class"]), src)
        CKPTS.setdefault(trainset_coll["id"], []).append(
            (load_network_cls(hp["network_class"]), tp, feature, [*tp.index.keys()], hp)
        )
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


client = storage.Client("ax6-Project")


@dtc.dataclass
class Checkpoint:
    id: str
    epoch: int
    root_dir: str = "./"
    bucket = "ax6-outputs"

    @staticmethod
    def get_id_and_epoch(path):
        id_, epoch = path.split("/")[-2:]
        return id_.strip("/"), int(epoch.split(".h5")[0].split("=")[-1])

    @staticmethod
    def from_blob(blob):
        path = blob.name
        id_, epoch = Checkpoint.get_id_and_epoch(path)
        ckpt = Checkpoint(id_, epoch)
        ckpt.bucket = blob.bucket.name
        return ckpt

    @staticmethod
    def from_path(path):
        basename = os.path.dirname(os.path.dirname(path))
        return Checkpoint(*Checkpoint.get_id_and_epoch(path), root_dir=basename)

    @property
    def gcp_path(self):
        return f"gs://{self.bucket}/checkpoints/{self.id}/epoch={self.epoch}.h5"

    @property
    def os_path(self):
        return os.path.join(self.root_dir, f"{self.id}/epoch={self.epoch}.h5")

    @property
    def blob(self):
        return client.bucket(self.bucket).blob(f"checkpoints/{self.id}/epoch={self.epoch}.h5")

    def download(self, force=False):
        os.makedirs(os.path.dirname(self.os_path), exist_ok=True)
        if os.path.isfile(self.os_path) and not force:
            return self
        try:
            client.download_blob_to_file(self.gcp_path, open(self.os_path, "wb"))
        except:
            self.delete()
            raise
        return self

    def delete(self):
        os.remove(self.os_path)

    @property
    def network(self):
        if not os.path.isfile(self.os_path):
            self.download()
        bank = CkptBank(self.os_path, 'r')
        hp = bank.ckpt.load_hp()
        return bank.ckpt.load_checkpoint(hp["cls"], "state_dict")

    @property
    def feature(self):
        if not os.path.isfile(self.os_path):
            self.download()
        bank = CkptBank(self.os_path, 'r')
        hp = bank.ckpt.load_hp()
        return hp['feature']

    @property
    def train_hp(self):
        return load_trainings_hp(os.path.join(self.root_dir, self.id))