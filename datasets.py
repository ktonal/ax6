import json
import dataclasses as dtc
from google.cloud import storage
import tempfile
import os

import h5mapper as h5m


client = storage.Client("ax6-Project")


def from_gcloud(feature):
    old_load = feature.load

    def new_load(self, source):
        with tempfile.NamedTemporaryFile() as f:
            client.download_blob_to_file(source, f)
            rv = old_load(f.name)
        return rv

    feature.load = type(old_load)(new_load, feature)
    return feature


def gcp_sound_bank(sr=16000):
    class GCPSoundBank(h5m.TypedFile):
        snd = from_gcloud(h5m.Sound(sr=sr, mono=True, normalize=True))
    return GCPSoundBank


def load_files(files, sr, filename="data.h5"):
    if "gs://" in files[0]:
        sb = gcp_sound_bank(sr)
        sb.create(filename, files, parallelism="threads", n_workers=8)
        return sb(filename, mode='r', keep_open=True)
    else:
        h5m.sound_bank.callback(filename, files[0], sr=sr)
        return h5m.TypedFile(filename, mode='r', keep_open=True)


@dtc.dataclass
class Trainset:
    keyword: str
    sr: int = 22050

    id = ""
    files = tuple()
    root_dir = "./"

    def __post_init__(self):
        collec = [c for c in TRAINSET if self.keyword in c["keywords"]]
        if len(collec) == 0:
            raise ValueError(f"keyword '{self.keyword}' couldn't be found in any trainset collections")
        self.id = collec[0]["id"]
        self.files = [f"gs://{b['bucket']}/{b['path']}" for b in collec[0]["blobs"]]

    @property
    def os_path(self):
        return os.path.join(self.root_dir, self.filename)

    @property
    def filename(self):
        return f"{self.keyword}.h5"

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        return load_files(self.files, self.sr, self.os_path)

    @property
    def bank(self):
        if not os.path.isfile(self.os_path):
            return self.download()
        return gcp_sound_bank(self.sr)(self.os_path, mode='r')


table = "trainset"
TRAINSET = [json.loads(blob.download_as_string())
            for blob in client.list_blobs(client.bucket("axx-data"), prefix=f"tables/{table}/collections")
            if blob.content_type == "application/json"]

VERDI_X = [c for c in TRAINSET if "Verdi_X" in c["keywords"]]
VERDI_X = {c["keywords"]: [f"gs://{b['bucket']}/{b['path']}" for b in c["blobs"]] for c in VERDI_X}

INSECTS_X = [c for c in TRAINSET if "Insects_X" in c["comments"]]
INSECTS_X = {c["keywords"]: [f"gs://{b['bucket']}/{b['path']}" for b in c["blobs"]] for c in INSECTS_X}

COUGH = [c for c in TRAINSET if "Cough" in c["keywords"]]
COUGH = {c["keywords"]: [f"gs://{b['bucket']}/{b['path']}" for b in c["blobs"]] for c in COUGH}
