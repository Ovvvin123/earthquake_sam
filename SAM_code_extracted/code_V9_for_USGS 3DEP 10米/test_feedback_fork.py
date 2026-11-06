import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint
from aio2_preview import preserve_rng


class FeedbackForkTests(unittest.TestCase):
    def test_fork_preserves_state_and_only_allows_declared_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = dict(resDir=str(root/'source'), emaTeacher=True, fixedVal=True,
                       onlineCorrection=True, unimatch=True, emaDecay=.98,
                       epochs=500, lossFunc='BCEFocalTversky', lr0=.0001)
            model = torch.nn.Linear(3, 1)
            opt = torch.optim.AdamW(model.parameters(), lr=.0001)
            teacher = EMATeacher(model, .98)
            model(torch.ones(2, 3)).sum().backward(); opt.step(); teacher.update(model)
            opt.zero_grad(set_to_none=True)
            path = root/'source.pth'
            save_ema_checkpoint(path, model, opt, teacher, 500, cfg)
            expected_rng = (random.random(), np.random.rand(), torch.rand(5))
            for mode in ('keep', 'disable'):
                dest = dict(cfg, resDir=str(root/mode), epochs=600, emaFeedbackFork=mode,
                            onlineCorrection=mode=='keep', unimatch=mode=='keep')
                other = torch.nn.Linear(3, 1)
                other_opt = torch.optim.AdamW(other.parameters(), lr=1.)
                other_teacher = EMATeacher(other)
                self.assertEqual(load_ema_checkpoint(path, other, other_opt, other_teacher, dest), 500)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, other.state_dict()[key]))
                for key, value in teacher.model.state_dict().items():
                    self.assertTrue(torch.equal(value, other_teacher.model.state_dict()[key]))
                self.assertEqual(other_teacher.decay, .98)
                self.assertEqual(other_teacher.num_updates, 1)
                self.assertEqual(opt.param_groups[0]['lr'], other_opt.param_groups[0]['lr'])
                for old, new in zip(opt.state.values(), other_opt.state.values()):
                    for key in old:
                        self.assertTrue(torch.equal(old[key], new[key]))
                self.assertEqual(random.random(), expected_rng[0])
                self.assertEqual(np.random.rand(), expected_rng[1])
                self.assertTrue(torch.equal(torch.rand(5), expected_rng[2]))
                for change in (dict(lr0=.1), dict(emaDecay=.9), dict(resDir=cfg['resDir']),
                               dict(unimatch=mode!='keep')):
                    with self.assertRaises(ValueError):
                        load_ema_checkpoint(path, other, other_opt, other_teacher, dict(dest, **change))
                (root/mode).mkdir()
                (root/mode/'model.pth').touch()
                with self.assertRaises(ValueError):
                    load_ema_checkpoint(path, other, other_opt, other_teacher, dest)
            with self.assertRaises(ValueError):
                load_ema_checkpoint(path, model, opt, teacher, dict(cfg, onlineCorrection=False, unimatch=False))

    def test_auxiliary_rng_isolation_keeps_data_randomness(self):
        torch.manual_seed(7); np.random.seed(7); random.seed(7)
        with preserve_rng():
            expected = (torch.rand(4), np.random.rand(), random.random())
        with preserve_rng():
            torch.manual_seed(501000007)
            torch.randn(2, 7, 16, 16).sum().item()
        self.assertTrue(torch.equal(torch.rand(4), expected[0]))
        self.assertEqual(np.random.rand(), expected[1]); self.assertEqual(random.random(), expected[2])


if __name__ == '__main__':
    unittest.main()
