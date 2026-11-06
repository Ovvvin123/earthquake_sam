"""Behavioral regression for collapse, unknown masks and corrected accumulation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import h5py
from fault_losses import (binary_fault_logit, fault_loss, supervision,
                          accumulation_window)
from fault_metrics import FaultMetrics
from aio2_online import fixed_validation, BestCheckpoints
from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint
from unimatch_fault import positive_consistency_logits
import test_aio2_ema as legacy


class LogitsModel(legacy.TinyModel):
    def forward(self, x, return_logits=False):
        z = self.backbone(self.head(x))
        return z if return_logits else z.softmax(1)


def options(**kw):
    args = dict(lossFunc='BCEWithLogits', tverskyWeight=.5, k0Only=True,
                swMode='single', isOther=False, supervisionKey='', collapseGuard=True)
    args.update(kw)
    return SimpleNamespace(**args)


class LossTests(unittest.TestCase):
    def test_probability_equivalence(self):
        z = torch.randn(2, 3, 7, 9) * 20
        torch.testing.assert_close(binary_fault_logit(z).sigmoid(), z.softmax(1)[:, :1])

    def test_saturated_background_and_positive_gradients_point_correctly(self):
        for value, target_value, sign in [(50., 0., 1), (-50., 1., -1)]:
            for name in ['BCEWithLogits', 'BCEFocalTversky']:
                z = torch.zeros(1, 3, 4, 4)
                z[:, 0] = value
                z.requires_grad_()
                y = torch.full_like(z[:, :1], target_value)
                loss = fault_loss(z, y, torch.ones_like(y), name)
                loss.backward()
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertTrue((sign*z.grad[:, 0] > .01).all())

    def test_unknown_pixels_have_zero_gradient_even_with_invalid_targets(self):
        for name in ['BCEWithLogits', 'BCEFocalTversky']:
            z = torch.randn(1, 3, 4, 4, requires_grad=True)
            y = torch.zeros_like(z[:, :1]); y[..., 2:] = float('nan')
            w = torch.ones_like(y); w[..., 2:] = 0
            loss = fault_loss(z, y, w, name)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(int(z.grad[..., 2:].count_nonzero()), 0)
            self.assertGreater(int(z.grad[..., :2].count_nonzero()), 0)

    def test_wholly_unknown_batch_is_zero_not_background(self):
        z = torch.randn(1, 3, 3, 3, requires_grad=True)
        y = torch.zeros_like(z[:, :1])
        loss = fault_loss(z, y, y, 'BCEFocalTversky')
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(int(z.grad.count_nonzero()), 0)

    def test_supervision_combines_known_and_input_validity(self):
        x = torch.ones(1, 7, 4, 4)
        d = torch.ones(1, 4, 4, 4)
        d[:, 0] = 0
        d[:, 3, :, 2:] = 0
        x[:, 2, 0] = 0
        y, w, _ = supervision(x, d, options(supervisionKey='known'), .01, .08, .1)
        self.assertEqual(int((w > 0).sum()), 6)
        self.assertEqual(int(y.sum()), 6)

    def test_tail_accumulation_matches_explicit_batch_means(self):
        a = torch.nn.Parameter(torch.tensor(1.))
        b = torch.nn.Parameter(a.detach().clone())
        oa, ob = torch.optim.SGD([a], lr=.01), torch.optim.SGD([b], lr=.01)
        data = torch.arange(1., 7.)
        steps = []
        for j, x in enumerate(data):
            divisor, update = accumulation_window(j, len(data), 4)
            ((a*x-2).square()/divisor).backward()
            if update:
                oa.step(); oa.zero_grad(); steps.append(j)
        for part in [data[:4], data[4:]]:
            ((b*part-2).square().mean()).backward()
            ob.step(); ob.zero_grad()
        self.assertEqual(steps, [3, 5])
        torch.testing.assert_close(a, b)
        self.assertIsNone(a.grad)

    def test_validation_teacher_logits_and_supervision_mask(self):
        model = LogitsModel(); teacher = EMATeacher(model)
        x = torch.ones(2, 7, 8, 8)
        d = torch.ones(2, 4, 8, 8); d[:, 0, :2] = 0; d[:, 3, :, 4:] = 0
        args = options(supervisionKey='known')
        values, teachers, health = fixed_validation(model, teacher, [(x, d)], 'cpu', args,
                                                     None, .01, .08, .1, return_metrics=True)
        self.assertEqual(values, teachers)
        self.assertEqual(health['student']['pixels'], 64)
        self.assertEqual(teacher.num_updates, 0)
        self.assertTrue(all(np.isfinite(list(health['baseline_losses'].values()))))

    def test_collapsed_checkpoint_rejected_but_diagnostics_kept(self):
        model = LogitsModel(); opt = torch.optim.SGD(model.parameters(), lr=.1)
        y = torch.zeros(1, 1, 4, 4); y[..., 0] = 1
        meter = FaultMetrics(); meter.observe(torch.ones_like(y), y, torch.ones_like(y))
        with TemporaryDirectory() as out:
            best = BestCheckpoints(out, {'collapseGuard': True})
            best.update(1, .1, None, model, opt, None, health={'student': meter.report()})
            self.assertIsNone(best.state['student'])
            self.assertFalse((Path(out)/'best_student/model.pth').exists())
            self.assertIn('all_positive', (Path(out)/'fixed_val.jsonl').read_text())

    def test_new_config_fields_cannot_change_on_resume(self):
        model = LogitsModel(); opt = torch.optim.AdamW(model.parameters()); teacher = EMATeacher(model)
        cfg = dict(lossFunc='BCEFocalTversky', tverskyWeight=.5,
                   accumulationVersion='window_v2', supervisionKey='known', supervisionVersion='known_mask_v2')
        with TemporaryDirectory() as out:
            path = Path(out)/'checkpoint.pth'
            save_ema_checkpoint(path, model, opt, teacher, 1, cfg)
            for key, value in [('lossFunc', 'BCEWithLogits'), ('tverskyWeight', 1.),
                               ('accumulationVersion', 'legacy_v1'), ('supervisionKey', '')]:
                with self.assertRaisesRegex(ValueError, 'differs'):
                    load_ema_checkpoint(path, model, opt, teacher, dict(cfg, **{key: value}))

    def test_positive_consistency_logits_stays_finite_and_masks_unknown(self):
        z = torch.zeros(1, 3, 4, 4); z[:, 0] = -100; z.requires_grad_()
        w = torch.zeros_like(z[:, :1]); w[..., 0] = .2
        loss = positive_consistency_logits(z, w)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((z.grad[:, 0, :, 0] < 0).all())
        self.assertEqual(int(z.grad[..., 1:].count_nonzero()), 0)

    def test_loader_mask_follows_crop_scale_and_d4(self):
        from tool import Generator_new
        with TemporaryDirectory() as out:
            path = Path(out)/'data.h5'
            mask = np.zeros((384, 384), dtype=np.float32); mask[:, :192] = 1
            with h5py.File(path, 'w') as f:
                g = f.create_group('tile')
                g['CTX'] = mask*255
                g['image'] = np.arange(384*384, dtype=np.float32).reshape(384,384)
                g['TW'] = np.stack([mask*255]*3)
                for key in ['k0', 'k1', 'k2']: g[key] = np.ones_like(mask)*.1
                g['known'] = mask
            for train in [False, True]:
                reader = Generator_new(str(path), ['tile'], training=train, fixed_crop=not train, supervision_key='known')
                for seed in range(8):
                    np.random.seed(seed)
                    x, d = reader[0]
                    self.assertTrue(np.isin(d[3], [0,1]).all())
                    # Linear image interpolation can differ at the boundary by at most one pixel.
                    self.assertLess(np.mean((x[0] > 0) != (d[3] > .5)), .01)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
