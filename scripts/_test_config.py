import sys
sys.path.insert(0, '.')
from scripts.run_batch import _load_config, batch_processor_from_config
from pathlib import Path

cfg = _load_config(Path('config/batch_processing.yaml'))
cfg['cellpose']['model_name_or_path'] = r'C:\Users\mzinn1\Desktop\CellposeTrainingDatasets\InVitroRGC_Snaps\models\cpdino_RGC-Snap'
cfg['cellpose'].pop('hf_repo_id', None)

proc = batch_processor_from_config(cfg, dry_run=True)
results = proc.run(Path(r'C:\Users\mzinn1\Desktop\Suite2pTest'))

from collections import Counter
counts = Counter(r['status'] for r in results)
print('Root:  C:\\Users\\mzinn1\\Desktop\\Suite2pTest')
print('Model: ' + proc.model_name_or_path)
for status, n in sorted(counts.items()):
    print(f'  {status}: {n}')
