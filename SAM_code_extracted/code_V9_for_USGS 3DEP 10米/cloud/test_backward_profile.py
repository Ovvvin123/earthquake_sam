"""Offline checks: bounded probe must honor the resolved cloud paths/config."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from check_v9_memory import load_probe_profile, probe_batch


class ProbeProfileTests(unittest.TestCase):
    def test_actual_batch_four_stacks_four_samples(self):
        import numpy as np
        class Reader:
            calls = []
            def __len__(self): return 2
            def __getitem__(self, index):
                self.calls.append(index)
                return np.full((7, 3, 3), index, dtype=np.float32), np.full((3, 3, 3), index+10, dtype=np.float32)
        reader = Reader()
        x, y = probe_batch(reader, 1, 4)
        self.assertEqual(reader.calls, [1, 0, 1, 0])
        self.assertEqual(x.shape, (4, 7, 3, 3))
        self.assertEqual(y.shape, (4, 3, 3, 3))
        np.testing.assert_array_equal(y[:, 0, 0, 0], [11, 10, 11, 10])
        with self.assertRaises(ValueError): probe_batch(reader, 0, 0)

    def test_cloud_batch_change_preserves_effective_batch_and_other_parameters(self):
        from run_experiment import read_profile
        root = Path(__file__).resolve().parent.parent
        local = read_profile(root/'experiments/california_v9_30m_1024_8gb.json')
        cloud = read_profile(root/'experiments/california_v9_30m_1024_cloud.json')
        self.assertEqual((cloud['batchsize'], cloud['stepPerMiniBatch']), (4, 2))
        self.assertEqual(cloud['batchsize']*cloud['stepPerMiniBatch'], local['batchsize']*local['stepPerMiniBatch'])
        allowed = {'batchsize', 'stepPerMiniBatch', 'dataset', 'dataFile', 'resDir', 'backupDir'}
        self.assertEqual({k:v for k,v in local.items() if k not in allowed},
                         {k:v for k,v in cloud.items() if k not in allowed})

    def test_resolved_profile_keeps_data_and_checkpoint_settings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = dict(dataset=str(root/'data/key_train.lst'), dataFile=str(root/'data/faults.h5'),
                        resDir=str(root/'runs/formal'), activationCheckpoint='decoder',
                        lrSchedule='warmup_hold_cosine', lrHoldEnd=100)
            profile = root/'resolved_cloud.json'
            profile.write_text(json.dumps({'arguments': args}))
            actual = load_probe_profile(root, Path('resolved_cloud.json'))
            for key in ('dataset', 'dataFile', 'activationCheckpoint', 'lrSchedule', 'lrHoldEnd'):
                self.assertEqual(actual[key], args[key])
            self.assertFalse((root/'runs').exists())
            override = load_probe_profile(root, profile, 'all')
            self.assertEqual(override['activationCheckpoint'], 'all')

    def test_missing_explicit_profile_never_falls_back_to_local(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_probe_profile(Path(tmp), Path('missing.json'))


if __name__ == '__main__':
    unittest.main()
