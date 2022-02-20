import json
import dataclasses as dtc
from google.cloud import storage
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def load_blobs(blobs, sr, filename="data.h5"):
    if "gs://" in blobs[0]:
        sb = gcp_sound_bank(sr)
        sb.create(filename, blobs, parallelism="threads", n_workers=8)
        return sb(filename, mode='r', keep_open=True)
    else:
        h5m.sound_bank.callback(filename, blobs[0], sr=sr)
        return h5m.TypedFile(filename, mode='r', keep_open=True)


def download_file(gcp_path, os_path):
    with open(os_path, "wb") as target:
        client.download_blob_to_file(gcp_path, target)
    return os_path


def download_blobs(blobs, target_root, force=False):
    targets = [os.path.join(target_root, os.path.split(f)[1]) for f in blobs]
    if not force:
        values = [(f, t) for f, t in zip(blobs, targets) if not os.path.isfile(t)]
        if not any(values):
            return targets
        blobs, targets = zip(*values)
    executor = ThreadPoolExecutor(max_workers=len(blobs))
    as_completed(executor.map(
        download_file, blobs, targets
    ))
    executor.shutdown(True)
    return targets


@dtc.dataclass
class Trainset:
    table: str
    keyword: str = ""
    id: str = ""
    sr: int = 22050

    blobs = list()
    files = list()
    root_dir = "./"
    tmp_cache = False

    def __post_init__(self):
        table = [json.loads(blob.download_as_string())
                 for blob in client.list_blobs(client.bucket("axx-data"), prefix=f"tables/{self.table}/collections")
                 if blob.content_type == "application/json"]
        collec = [c for c in table
                  if (self.keyword and self.keyword in c["keywords"]) or (self.id and self.id in c["id"])]
        if len(collec) == 0:
            raise ValueError(f"neither keyword '{self.keyword}' nor id `{self.id}`"
                             f" could be found in {self.table}")
        self.id = collec[0]["id"]
        self.blobs = [f"gs://{b['bucket']}/{b['path']}" for b in collec[0]["blobs"]]
        self.files = [os.path.join(self.root_dir, self.keyword, os.path.split(f)[1])
                      for f in self.blobs]

    @property
    def os_path(self):
        return os.path.join(self.root_dir, self.keyword or self.id[:12], self.filename)

    @property
    def filename(self):
        return f"{self.keyword or self.id[:12]}_{self.sr}.h5"

    @property
    def bank(self):
        if not os.path.isfile(self.os_path):
            return self.download()
        return h5m.TypedFile(self.os_path, mode='r', keep_open=True)

    def download(self, force=False):
        if force or not os.path.isfile(self.os_path):
            if self.tmp_cache:
                os.makedirs(self.root_dir, exist_ok=True)
                return load_blobs(self.blobs, self.sr, self.os_path)
            target_dir = os.path.dirname(self.os_path)
            os.makedirs(target_dir, exist_ok=True)
            targets = download_blobs(self.blobs, target_dir, force)
            h5m.sound_bank.callback(self.os_path, targets, sr=self.sr, parallelism="mp", n_workers=len(targets))
        return self

    def delete(self):
        os.remove(self.os_path)

    def upload(self):
        pass


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

LUNGS = [c for c in TRAINSET if "Lung Collection" in c["keywords"]]
LUNGS = {c["keywords"]: [f"gs://{b['bucket']}/{b['path']}" for b in c["blobs"]] for c in LUNGS}
