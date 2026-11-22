"""Create an ignored, auditable TF2-compat copy; never modify upstream."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

COMMIT = 'c010dd74ff4b7566bd0ac989c90a32cfbc630d84'


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT:
        raise ValueError('Unexpected upstream revision')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Upstream tracked files must be clean')
    destination.mkdir(parents=True, exist_ok=False)
    changes = []
    for path in sorted(source.rglob('*')):
        if '.git' in path.parts or path.suffix not in {'.py', '.npy'}:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.npy':
            shutil.copyfile(path, target)
            continue
        original = path.read_text()
        updated = original.replace('import tensorflow as tf', 'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()')
        updated = updated.replace('import tensorflow.contrib.layers as tl', 'import compat_layers as tl')
        updated = updated.replace('dtype=np.bool)', 'dtype=bool)')
        updated = updated.replace('ops.convert_to_tensor(', 'tf.convert_to_tensor(')
        target.write_text(updated)
        if original != updated:
            changes.append({'file': str(relative), 'before_sha256': hashlib.sha256(original.encode()).hexdigest(),
                            'after_sha256': hashlib.sha256(updated.encode()).hexdigest()})
    shutil.copyfile(Path(__file__).with_name('compat_layers.py'), destination/'compat_layers.py')
    (destination/'compat_manifest.json').write_text(json.dumps({'commit': revision, 'changes': changes, 'adapter_sha256': hashlib.sha256((destination/'compat_layers.py').read_bytes()).hexdigest()}, indent=2)+'\n')
    print('Prepared', destination, 'with', len(changes), 'modified source files')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--destination', required=True)
    a = p.parse_args()
    prepare(a.source, a.destination)
