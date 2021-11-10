import json
from google.cloud import storage
import io
import tempfile

import h5mapper as h5m


client = storage.Client("ax6-Project")


def from_gcloud(feature):
    old_load = feature.load

    def new_load(self, source):
        with tempfile.NamedTemporaryFile() as f:
            client.download_blob_to_file(source, f)
            return old_load(f.name)

    feature.load = type(old_load)(new_load, feature)
    return feature


def gcp_sound_bank(sr=16000):
    class GCPSoundBank(h5m.TypedFile):
        snd = from_gcloud(h5m.Sound(sr=sr, mono=True, normalize=True))
    return GCPSoundBank


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
