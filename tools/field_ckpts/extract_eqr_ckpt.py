"""Restricted extraction of locuslab/EqR-model sudoku-extreme/eqr.pth WITHOUT torch.load(weights_only=False).
The pickle was scanned with pickletools (2026-09-05): its only GLOBAL references are
  __builtin__ {dict,list,long}, collections {OrderedDict,defaultdict}, typing Any,
  omegaconf.base {ContainerMetadata,Metadata}, omegaconf.dictconfig DictConfig, omegaconf.listconfig ListConfig,
  omegaconf.nodes AnyNode, torch FloatStorage, torch._utils _rebuild_tensor_v2.
This unpickler resolves ONLY those names: omegaconf classes become inert stubs (no library code runs), tensors are
rebuilt from the zip's raw storage bytes in numpy. Anything else raises. Output: eqr_weights.npz + eqr_config.json."""
import collections, io, json, pickle, sys, typing, zipfile
from pathlib import Path
import numpy as np

SRC = Path(sys.argv[1]); OUT = SRC.parent / "eqr_extracted"; OUT.mkdir(exist_ok=True)
zf = zipfile.ZipFile(SRC); names = zf.namelist(); root = names[0].split("/")[0]
pkl = zf.read(f"{root}/data.pkl")

class Stub:
    def __init__(self, *a, **k): self.args = a; self.state = None
    def __setstate__(self, st): self.state = st
class DictConfigStub(Stub): pass
class ListConfigStub(Stub): pass
class AnyNodeStub(Stub): pass
class MetaStub(Stub): pass
class StorageRef:                       # ('storage', cls, key, location, numel)
    def __init__(self, key, numel, dtype): self.key, self.numel, self.dtype = key, numel, dtype
DTYPES = {"FloatStorage": np.float32, "BFloat16Storage": None, "HalfStorage": np.float16, "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_, "ByteStorage": np.uint8}
class StorageCls:
    def __init__(self, name): self.name = name

def rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    raw = zf.read(f"{root}/data/{storage.key}")
    if storage.dtype is None:  # bfloat16: upcast via uint16 -> float32
        u = np.frombuffer(raw, dtype=np.uint16, count=storage.numel).astype(np.uint32) << 16
        arr = u.view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=storage.dtype, count=storage.numel)
    itemsize = arr.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride))
    return np.array(view)  # materialized copy

ALLOW = {("collections", "OrderedDict"): collections.OrderedDict, ("collections", "defaultdict"): collections.defaultdict,
         ("__builtin__", "dict"): dict, ("__builtin__", "list"): list, ("__builtin__", "long"): int, ("typing", "Any"): typing.Any,
         ("omegaconf.base", "ContainerMetadata"): MetaStub, ("omegaconf.base", "Metadata"): MetaStub,
         ("omegaconf.dictconfig", "DictConfig"): DictConfigStub, ("omegaconf.listconfig", "ListConfig"): ListConfigStub,
         ("omegaconf.nodes", "AnyNode"): AnyNodeStub, ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor_v2}
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch" and name.endswith("Storage"): return StorageCls(name)
        if (module, name) in ALLOW: return ALLOW[(module, name)]
        raise pickle.UnpicklingError(f"REFUSED global {module}.{name}")
    def persistent_load(self, pid):
        assert isinstance(pid, tuple) and pid[0] == "storage", pid
        _, cls, key, location, numel = pid
        return StorageRef(str(key), int(numel), DTYPES[cls.name])
obj = RestrictedUnpickler(io.BytesIO(pkl)).load()

def plain(x):
    if isinstance(x, DictConfigStub): return {k: plain(v) for k, v in x.state["_content"].items()} if isinstance(x.state.get("_content"), dict) else None
    if isinstance(x, ListConfigStub): return [plain(v) for v in x.state["_content"]] if isinstance(x.state.get("_content"), list) else None
    if isinstance(x, AnyNodeStub): return plain(x.state["_val"])
    if isinstance(x, (dict, collections.OrderedDict)): return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return f"<tensor {x.shape} {x.dtype}>"
    if isinstance(x, Stub): return f"<stub {type(x).__name__}>"
    return x

print("top-level keys:", list(obj.keys()))
weights = {}
for section in ("model", "ema"):
    v = obj.get(section)
    if v is None: print(f"{section}: absent"); continue
    sd = v.get("shadow", v) if section == "ema" and isinstance(v, dict) and "shadow" in v else v
    n = 0
    for k, t in sd.items():
        if isinstance(t, np.ndarray): weights[f"{section}/{k}"] = t; n += t.size
        else: print(f"   {section}/{k}: {plain(t)}")
    print(f"{section}: {len([k for k in weights if k.startswith(section)])} tensors, {n} params; mu =", v.get("mu") if isinstance(v, dict) else None)
for k in list(weights)[:40]: print(f"   {k:60s} {weights[k].shape} {weights[k].dtype}")
np.savez(OUT / "eqr_weights.npz", **weights)
cfg = plain(obj.get("config")); meta = {k: plain(v) for k, v in obj.items() if k not in ("model", "ema", "config", "optimizers", "carry", "rng", "numpy", "python", "train_dataset")}
json.dump(dict(config=cfg, meta=meta), open(OUT / "eqr_config.json", "w"), indent=1, default=str)
print("config.arch:", json.dumps(cfg.get("arch") if cfg else None, indent=1)); print("meta:", json.dumps(meta, default=str)[:800])
print("saved", OUT)
