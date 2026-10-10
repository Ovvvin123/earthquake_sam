"""Tests for actual-loop integration, resume and incomplete-label semantics."""
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import test_aio2_ema as ema_tests
from test_aio2_online import options as online_options
from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint
from unimatch_fault import (strong_view, complementary_masks, reliable_weights,
                           positive_consistency, backward_strong_views,
                           consistency_weight, ConsistencyDiagnostics)


def options(**overrides):
    values = vars(online_options(onlineStart=1, onlineHoldEnd=1, onlineRampEnd=3,
                                onlineBuffer=0, onlineMinPixels=1))
    values.update(unimatch=True, unimatchStart=1, unimatchRampEnd=3,
                  unimatchWeight=.1, unimatchNoDropout=False,
                  unimatchContrast=.2, unimatchBrightness=.1, unimatchNoise=.03)
    values.update(overrides)
    return SimpleNamespace(**values)


class MaskModel(ema_tests.TinyModel):
    def complementary_feature_channels(self):
        return (3,)

    def forward(self, x, feature_masks=None):
        feature = self.head(x)
        if feature_masks is not None:
            feature = feature * feature_masks[0]
        return self.backbone(feature).sigmoid()


class BrokenStrongModel(MaskModel):
    def forward(self, x, feature_masks=None):
        p = super().forward(x, feature_masks)
        return p * float('nan') if feature_masks is not None else p


class UniMatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.image = torch.rand(2, 7, 8, 8) * 2 - 1
        self.image[:, 2:4] = 1
        self.image[:, 2:4, :2, :2] = 0
        self.original = torch.zeros(2, 1, 8, 8)
        self.original[:, :, 3, 3] = 1
        self.pseudo = torch.zeros_like(self.original, dtype=torch.bool)
        self.pseudo[:, :, 5:7, 5:7] = True

    def test_schedule_and_zero_mode(self):
        args = options(unimatchStart=41, unimatchRampEnd=80)
        for epoch, value in [(40, 0), (41, .0025), (60, .05), (80, .1), (200, .1)]:
            self.assertAlmostEqual(consistency_weight(epoch, args), value)
        self.assertEqual(consistency_weight(100, options(unimatch=False)), 0)

    def test_augmentation_preserves_depth_masks_invalid_pixels_and_source(self):
        before = self.image.clone()
        first, second = strong_view(self.image, options()), strong_view(self.image, options())
        self.assertTrue(torch.equal(self.image, before))
        for output in (first, second):
            self.assertTrue(torch.equal(output[:, 1:4], before[:, 1:4]))
            self.assertTrue(torch.equal(output[:, :, :2, :2], before[:, :, :2, :2]))
        self.assertFalse(torch.equal(first, second))
        rng = torch.get_rng_state()
        same = strong_view(self.image, options(unimatchContrast=0, unimatchBrightness=0, unimatchNoise=0))
        self.assertTrue(torch.equal(same, before))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_masks_are_complementary_at_every_scale(self):
        first, second = complementary_masks(2, (256, 64, 32, 32), 'cpu', torch.float32)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a + b, torch.full_like(a, 2)))
            self.assertTrue(torch.equal(a * b, torch.zeros_like(a)))
            self.assertEqual(a.shape[-2:], (1, 1))

    def test_no_background_gradient_and_candidate_discount_does_not_cancel(self):
        original = self.original.clone()
        original[:, :, 0, 0] = 1  # invalid positive must be ignored
        weights = reliable_weights(original, self.pseudo, self.image, .2)
        self.assertEqual(float(weights[:, :, 0, 0].sum()), 0)
        p = torch.full_like(original, .4, requires_grad=True)
        loss = positive_consistency(p, weights)
        loss.backward()
        self.assertTrue((p.grad[weights > 0] < 0).all())
        self.assertTrue((p.grad[weights == 0] == 0).all())
        candidate_weights = reliable_weights(torch.zeros_like(original), self.pseudo, self.image, .2)
        full_weights = reliable_weights(torch.zeros_like(original), self.pseudo, self.image, 1)
        torch.testing.assert_close(positive_consistency(p, candidate_weights),
                                   .2 * positive_consistency(p, full_weights))
        self.assertFalse(weights.requires_grad)

    def test_no_candidates_or_zero_weight_skips_strong_forwards_and_rng(self):
        model = MaskModel()
        model.forward = lambda *a, **kw: self.fail('No strong forward expected')
        with TemporaryDirectory() as directory:
            for weight, original, pseudo in [
                    (.1, torch.zeros_like(self.original), torch.zeros_like(self.pseudo)),
                    (0., self.original, self.pseudo)]:
                rng = torch.get_rng_state()
                stats, sample = backward_strong_views(
                    model, self.image, original, pseudo, .2, options(), weight,
                    4, directory, 1, 0, capture=True)
                self.assertEqual(stats['active'], 0)
                self.assertIsNone(sample)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_dual_backward_diagnostics_and_finite_zero_probabilities(self):
        with TemporaryDirectory() as directory:
            model = MaskModel()
            stats, sample = backward_strong_views(
                model, self.image, self.original, self.pseudo, .2,
                options(), .1, 4, directory, 1, 0, capture=True)
            self.assertEqual(stats['active'], 1)
            self.assertTrue(torch.isfinite(model.head[0].weight.grad).all())
            self.assertGreater(float(model.head[0].weight.grad.abs().sum()), 0)
            logger = ConsistencyDiagnostics(1, .1, directory, capture=True)
            logger.observe(.5, stats, sample)
            report = logger.save()
            self.assertAlmostEqual(report['total_loss'], .5 + .1 * stats['strong_loss'])
            self.assertTrue((Path(directory) / 'unimatch/epoch_0001.png').exists())
            self.assertTrue(torch.equal(torch.from_numpy(sample['strong1'][1:4]),
                                        torch.from_numpy(sample['weak'][1:4])))
            p = torch.zeros_like(self.original, requires_grad=True)
            loss = positive_consistency(p, self.original)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_nonfinite_strong_prediction_is_reported(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FloatingPointError):
                backward_strong_views(BrokenStrongModel(), self.image, self.original,
                                      self.pseudo, .2, options(), .1, 4, directory, 2, 3)
            report = json.loads((Path(directory) / 'numerical_failure.json').read_text())
            self.assertEqual(report['view'], 1)
            self.assertEqual(report['batch'], 3)

    def test_actual_training_loop_resume_exact_and_zero_weight_matches_disabled(self):
        runner = ema_tests.EMATests()
        runner.setUp()
        initial = MaskModel()
        image = self.image[:1]
        distances = torch.ones(1, 3, 8, 8)
        distances[:, :, 3, 3] = 0
        batches = [(image.clone(), distances.clone()) for _ in range(6)]
        args = vars(options())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            whole, teacher = runner._run_training(initial, root/'whole', 3,
                arg_overrides=args, batches_override=batches)
            runner._run_training(initial, root/'split', 1,
                arg_overrides=args, batches_override=batches)
            resumed, resumed_teacher = runner._run_training(initial, root/'split', 3,
                resume=str(root/'split/ema_checkpoint.pth'),
                arg_overrides=args, batches_override=batches)
            self.assertEqual(teacher.num_updates, 6)  # not tripled by strong views
            for key, value in whole.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            for key, value in teacher.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed_teacher.model.state_dict()[key]), key)
            disabled, _ = runner._run_training(initial, root/'disabled', 2,
                arg_overrides=dict(args, unimatch=False), batches_override=batches)
            zero, _ = runner._run_training(initial, root/'zero', 2,
                arg_overrides=dict(args, unimatchWeight=0), batches_override=batches)
            for key, value in disabled.state_dict().items():
                self.assertTrue(torch.equal(value, zero.state_dict()[key]), key)
            report = json.loads((root/'whole/unimatch/epoch_0001.json').read_text())
            self.assertEqual(report['active_batches'], 6)
            self.assertGreater(report['weighted_strong_loss'], 0)

    def test_cli_validation_and_existing_output_protection(self):
        import subprocess
        import sys
        with TemporaryDirectory() as directory:
            command = [sys.executable, str(Path(__file__).with_name('config.py')),
                       '--unimatch', '--onlineCorrection', '--emaTeacher',
                       '--k0Only', '--lossFunc', 'FocalTversky', '--pu', '0',
                       '--resDir', directory]
            good = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            for extra in (['--pu', '.4'], ['--unimatchWeight', 'nan'],
                          ['--unimatchNoise', '-.1'], ['--unimatchStart', '1'],
                          ['--isalign']):
                bad = subprocess.run(command + extra, capture_output=True, text=True)
                self.assertEqual(bad.returncode, 2, str(extra))
            Path(directory, 'model.pth').touch()
            bad = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertIn('already contains a run', bad.stderr)

    def test_legacy_resume_and_new_config_guards(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = MaskModel()
            opt = torch.optim.AdamW(model.parameters())
            teacher = EMATeacher(model)
            old = dict(resDir=str(root/'warm'), emaDecay=.99, pu=0.)
            save_ema_checkpoint(root/'old.pth', model, opt, teacher, 40, old)
            self.assertEqual(load_ema_checkpoint(root/'old.pth', model, opt, teacher,
                dict(old, unimatch=False, unimatchStart=41, unimatchWeight=.1)), 40)
            new = dict(old, **vars(options(onlineStart=41, onlineHoldEnd=50,
                onlineRampEnd=80, unimatchStart=41, unimatchRampEnd=80)))
            new.update(emaStageTransition=True, resDir=str(root/'new'), emaDecay=.98)
            self.assertEqual(load_ema_checkpoint(root/'old.pth', model, opt, teacher, new), 40)
            save_ema_checkpoint(root/'new.pth', model, opt, teacher, 41, new)
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_ema_checkpoint(root/'new.pth', model, opt, teacher,
                                    dict(new, emaStageTransition=False, unimatchWeight=.9))
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_ema_checkpoint(root/'new.pth', model, opt, teacher,
                                    dict(new, emaStageTransition=False, unimatch=False))


if __name__ == '__main__':
    unittest.main()
