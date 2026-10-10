"""Behavioral tests for conservative online correction and the actual train branch."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import test_aio2_ema as ema_tests
from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint
from aio2_online import (pseudo_weight, select_candidates, temporary_targets,
                         fixed_validation, BestCheckpoints, validation_manifest)


def options(**overrides):
    defaults = dict(onlineCorrection=True, onlineStart=41, onlineHoldEnd=50,
                    onlineRampEnd=80, onlineWeightStart=.05, onlineWeightMax=.2,
                    onlineThreshold=.9, onlineSupport=.5, onlineBuffer=1,
                    onlineMinPixels=2, onlineMaxFraction=.2, fixedVal=True,
                    checkpointEvery=2, emaStageTransition=False)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PatternModel(ema_tests.TinyModel):
    def forward(self, x):
        # A spatially isolated high probability patch with a trainable logit offset.
        offset = self.head[0].weight.mean()
        return torch.sigmoid(10 * x[:, :1] - 5 + offset).expand(-1, 3, -1, -1)


class OnlineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.loss = ema_tests._function_from_file('torchUnet.py', 'FocalTversky', {'torch': torch})

    def test_schedule_boundaries(self):
        args = options()
        expected = {1: 0, 40: 0, 41: .05, 50: .05, 51: .055, 65: .125,
                    80: .2, 100: .2, 200: .2}
        for epoch, value in expected.items():
            self.assertAlmostEqual(pseudo_weight(epoch, args), value)

    def test_reject_expansion_before_cutting_out_original_and_keep_separate_object(self):
        p = np.zeros((20, 20))
        truth = np.zeros_like(p, dtype=bool)
        truth[2, 2] = True
        p[1:6, 1:6] = .95  # big high-confidence ring connected to original
        p[12:15, 12:15] = .95
        chosen, stats = select_candidates(p, truth, np.ones_like(truth), options())
        self.assertFalse(chosen[:8, :8].any())
        self.assertEqual(chosen.sum(), 9)
        self.assertEqual(stats['rejected_touching'], 1)
        # A medium-confidence bridge must also prevent accepting a label expansion.
        p[5:14, 5] = .6
        p[13, 5:13] = .6
        chosen, _ = select_candidates(p, truth, np.ones_like(truth), options())
        self.assertFalse(chosen.any())

    def test_invalid_small_nonfinite_and_area_budget(self):
        p = np.zeros((20, 20))
        truth = np.zeros_like(p, dtype=bool)
        valid = np.ones_like(truth)
        p[2, 2] = .95  # too small
        p[5:7, 5:7] = .95
        valid[5:7, 5:7] = False
        p[10:12, 10:12] = np.nan
        p[15:17, 15:17] = .8  # no high-confidence seed
        chosen, _ = select_candidates(p, truth, valid, options())
        self.assertFalse(chosen.any())
        p[:] = .95
        chosen, stats = select_candidates(p, truth, valid, options())
        self.assertFalse(chosen.any())
        self.assertGreater(stats['rejected_size'], 0)
        p[:] = 0
        p[2:4, 2:5] = .95
        p[10:12, 10:13] = .95
        chosen, stats = select_candidates(p, truth, np.ones_like(truth),
                                           options(onlineMaxFraction=.02))
        self.assertFalse(chosen.any())
        self.assertEqual(stats['rejected_budget'], 2)

    def test_temporary_supervision_gradient_and_identity_without_candidates(self):
        target = torch.zeros(1, 1, 20, 20)
        target[:, :, 2, 2] = 1
        sw = torch.ones_like(target)
        sw[:, :, 12:15, 12:15] = .1  # far from original annotation
        teacher_p = torch.zeros_like(target, requires_grad=True)
        with torch.no_grad():
            teacher_p[:, :, 12:15, 12:15] = .95
        image = torch.ones(1, 7, 20, 20)
        corrected, weights, mask, _ = temporary_targets(target, sw, teacher_p, image, .2, options())
        self.assertEqual(int(mask.sum()), 9)
        self.assertTrue(torch.equal(corrected[target.bool()], target[target.bool()]))
        self.assertTrue(torch.equal(weights[~mask], sw[~mask]))
        torch.testing.assert_close(weights[mask], torch.full((9,), .2))
        prediction = torch.full_like(target, .4, requires_grad=True)
        base = self.loss(target, prediction, sw=sw, pu=0)
        base_grad = torch.autograd.grad(base, prediction)[0]
        corrected_loss = self.loss(corrected, prediction, sw=weights, pu=0)
        gradient = torch.autograd.grad(corrected_loss, prediction)[0]
        self.assertTrue((base_grad[mask] > 0).all())  # old supervision suppressed candidates
        self.assertTrue((gradient[mask] < 0).all())   # now encourages positives, no conflicting FP
        self.assertIsNone(teacher_p.grad)
        self.assertFalse(corrected.requires_grad)
        self.assertEqual(int(target.sum()), 1)       # original never edited
        absent = torch.zeros_like(teacher_p)
        same_target, same_sw, new_mask, _ = temporary_targets(target, sw, absent, image, .2, options())
        self.assertIs(same_target, target)
        self.assertIs(same_sw, sw)
        self.assertFalse(new_mask.any())             # previous candidate not retained
        same_target, same_sw, mask0, _ = temporary_targets(target, sw, teacher_p, image, 0, options())
        self.assertIs(same_target, target)
        self.assertIs(same_sw, sw)
        self.assertFalse(mask0.any())

    def test_fixed_validation_repeatable_preserves_rng_modes_and_uses_all_samples(self):
        model = PatternModel()
        model.train()
        teacher = EMATeacher(model)
        args = SimpleNamespace(k0Only=True, swMode='single', isOther=False, lossFunc='FocalTversky')
        images = torch.ones(3, 7, 8, 8)
        distances = torch.ones(3, 3, 8, 8)
        distances[0, :, 1:3] = 0
        distances[2, :, 5:7] = 0
        modes = [m.training for m in model.modules()]
        rng = torch.get_rng_state().clone()
        grouped = [(images[:2], distances[:2]), (images[2:], distances[2:])]
        a = fixed_validation(model, teacher, grouped, 'cpu', args, self.loss, .02, .08, .1)
        b = fixed_validation(model, teacher, [(images, distances)], 'cpu', args, self.loss, .02, .08, .1)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 3)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(teacher.num_updates, 0)

    def test_fixed_validation_fingerprint_detects_changed_data(self):
        with TemporaryDirectory() as directory:
            image = torch.ones(1, 7, 8, 8)
            target = torch.zeros(1, 3, 8, 8)
            loader = [(image, target)]
            signature = validation_manifest(loader, directory)
            self.assertEqual(signature, validation_manifest(loader, directory))
            target[0, 0, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, 'samples changed'):
                validation_manifest(loader, directory)
            state = {'student': None, 'teacher': None, 'validation_signature': signature}
            with self.assertRaisesRegex(ValueError, 'differs'):
                BestCheckpoints(directory, {}, state, signature='different')

    def test_stage_transition_and_strict_normal_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = PatternModel()
            opt = torch.optim.AdamW(model.parameters(), lr=.0123)
            teacher = EMATeacher(model, .99)
            teacher.num_updates = 800
            original = {'resDir': str(root / 'warm'), 'emaDecay': .99, 'pu': 0.}
            path = root / 'warm.pth'
            save_ema_checkpoint(path, model, opt, teacher, 40, original)
            config = dict(original)
            config.update(vars(options()))
            config.update(resDir=str(root / 'online'), emaDecay=.98, emaStageTransition=True)
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_ema_checkpoint(path, model, opt, teacher, dict(config, emaStageTransition=False))
            self.assertEqual(load_ema_checkpoint(path, model, opt, teacher, config), 40)
            self.assertEqual(teacher.decay, .98)
            self.assertEqual(teacher.num_updates, 800)
            self.assertEqual(opt.param_groups[0]['lr'], .0123)
            with self.assertRaisesRegex(ValueError, 'NEW'):
                load_ema_checkpoint(path, model, opt, teacher, dict(config, resDir=original['resDir']))
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_ema_checkpoint(path, model, opt, teacher, dict(config, pu=.4))
            save_ema_checkpoint(root / 'online.pth', model, opt, teacher, 41, config)
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_ema_checkpoint(root / 'online.pth', model, opt, teacher,
                                    dict(config, emaStageTransition=False, onlineWeightMax=.5))
            resumed = dict(config, emaStageTransition=False)
            self.assertEqual(load_ema_checkpoint(root / 'online.pth', model, opt, teacher, resumed), 41)

    def test_best_checkpoint_survives_worse_epoch_and_restores_selection_state(self):
        with TemporaryDirectory() as directory:
            model = PatternModel()
            teacher = EMATeacher(model)
            opt = torch.optim.AdamW(model.parameters())
            manager = BestCheckpoints(directory, {})
            manager.update(40, .4, .5, model, opt, teacher, baseline=True)
            first = torch.load(Path(directory) / 'best_student/model.pth', weights_only=True)
            with torch.no_grad():
                model.head[0].weight.add_(1)
            manager.update(41, .6, .3, model, opt, teacher)
            second = torch.load(Path(directory) / 'best_student/model.pth', weights_only=True)
            for key in first:
                self.assertTrue(torch.equal(first[key], second[key]))
            saved = torch.load(Path(directory) / 'best_teacher/ema_checkpoint.pth', weights_only=True)
            self.assertEqual(saved['training_state']['student']['epoch'], 40)
            self.assertEqual(saved['training_state']['teacher']['epoch'], 41)
            self.assertEqual(saved['next_epoch'], 41)

    def test_actual_loop_online_changes_student_and_resume_is_exact(self):
        args = vars(options(onlineStart=1, onlineHoldEnd=1, onlineRampEnd=3,
                            onlineMinPixels=1, onlineBuffer=0, onlineMaxFraction=.5,
                            previewEvery=99))
        image = torch.zeros(1, 7, 8, 8)
        image[:, 2:4] = 1
        image[:, 0, 5:7, 5:7] = 1
        target = torch.ones(1, 3, 8, 8)
        target[:, :, 0, 0] = 0
        batches = [(image.clone(), target.clone()) for _ in range(6)]
        runner = ema_tests.EMATests()
        runner.setUp()
        initial = PatternModel()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            whole, teacher = runner._run_training(
                initial, root / 'whole', 3, arg_overrides=args, batches_override=batches)
            runner._run_training(initial, root / 'split', 1, arg_overrides=args, batches_override=batches)
            resumed, resumed_teacher = runner._run_training(
                initial, root / 'split', 3, resume=str(root / 'split/ema_checkpoint.pth'),
                arg_overrides=args, batches_override=batches)
            for key, tensor in whole.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.state_dict()[key]), key)
            for key, tensor in teacher.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed_teacher.model.state_dict()[key]), key)
            baseline, _ = runner._run_training(
                initial, root / 'baseline', 3, arg_overrides=dict(args, onlineCorrection=False),
                batches_override=batches)
            self.assertFalse(torch.equal(whole.head[0].weight, baseline.head[0].weight))
            report = json.loads((root / 'whole/online/epoch_0001/summary.json').read_text())
            self.assertGreater(report['totals']['candidate_pixels'], 0)
            self.assertTrue((root / 'whole/checkpoints/epoch_0002.pth').exists())
            self.assertTrue((root / 'whole/best_student/model.pth').exists())
            self.assertTrue((root / 'whole/online/epoch_0001/batch_sample_00.png').exists())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
