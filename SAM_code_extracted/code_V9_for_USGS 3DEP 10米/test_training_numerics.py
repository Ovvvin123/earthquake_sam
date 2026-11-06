"""Regression for the epoch-158 zero-loss / NaN-gradient failure."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_aio2_ema as ema_tests
from aio2_ema import EMATeacher
from aio2_online import fixed_validation
from training_numerics import clip_finite_gradients


class BadGradientModel(ema_tests.TinyModel):
    def forward(self, x):
        prediction = super().forward(x)
        if prediction.requires_grad:
            prediction.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
        return prediction


class BadPredictionModel(ema_tests.TinyModel):
    def forward(self, x):
        return super().forward(x) * float('nan')


class NumericsTests(unittest.TestCase):
    def setUp(self):
        self.loss = ema_tests._function_from_file('torchUnet.py', 'FocalTversky', {'torch': torch})

    def test_empty_perfect_and_rounded_perfect_gradients_are_finite_with_or_without_pu(self):
        for pu in (0., .4):
            for target_value, prediction_value in ((0., 0.), (0., 1e-20), (1., 1.)):
                prediction = torch.full((2, 1, 16, 16), prediction_value, requires_grad=True)
                target = torch.full_like(prediction, target_value)
                loss = self.loss(target, prediction, pu=pu)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(prediction.grad).all())
                self.assertLess(float(loss), 1e-5)

    def test_ordinary_loss_and_gradient_match_previous_formula_exactly(self):
        def old_loss(target, prediction, sw, pu):
            fp_w = sw * (1 - min(pu, .6) * prediction.detach().clamp(0, 1))
            tp = (sw * target * prediction).sum()
            fn = (sw * target * (1 - prediction)).sum()
            fp = (fp_w * (1 - target) * prediction).sum()
            return (1 - (tp + 1e-5) / (tp + .7 * fn + .3 * fp + 1e-5)) ** .75
        torch.manual_seed(3)
        for pu in (0., .4):
            target = (torch.rand(2, 1, 8, 8) > .8).float()
            prediction = (.1 + .8 * torch.rand_like(target)).requires_grad_()
            sw = .1 + torch.rand_like(target)
            expected = old_loss(target, prediction, sw, pu)
            observed = self.loss(target, prediction, sw=sw, pu=pu)
            self.assertTrue(torch.equal(expected, observed))
            a = torch.autograd.grad(expected, prediction)[0]
            b = torch.autograd.grad(observed, prediction)[0]
            self.assertTrue(torch.equal(a, b))

    def test_invalid_gradient_blocks_actual_training_optimizer_step(self):
        runner = ema_tests.EMATests()
        runner.setUp()
        with TemporaryDirectory() as directory:
            with patch('torch.optim.AdamW.step', side_effect=AssertionError('Invalid optimizer step reached')):
                with self.assertRaisesRegex(FloatingPointError, 'gradient norm'):
                    runner._run_training(BadGradientModel(), directory, 1)
            report = json.loads((Path(directory) / 'numerical_failure.json').read_text())
            self.assertEqual(report['epoch'], 1)
            self.assertEqual(report['batch'], 0)
            self.assertTrue(report['parameters'])
            self.assertFalse((Path(directory) / 'ema_checkpoint.pth').exists())

    def test_validation_names_failed_teacher_sample_and_restores_modes(self):
        student = ema_tests.TinyModel().train()
        teacher = EMATeacher(BadPredictionModel())
        image = torch.ones(1, 7, 8, 8)
        distances = torch.zeros(1, 3, 8, 8)
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(resDir=directory, k0Only=True, swMode='single',
                                   isOther=False, lossFunc='FocalTversky')
            with self.assertRaisesRegex(FloatingPointError, 'teacher validation prediction'):
                fixed_validation(student, teacher, [(image, distances)], 'cpu', args,
                                 self.loss, .02, .08, .1, epoch=158)
            report = json.loads((Path(directory) / 'numerical_failure.json').read_text())
            self.assertEqual(report['epoch'], 158)
            self.assertEqual(report['validation_sample'], 0)
            self.assertTrue(student.training)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_near_zero_loss_backward_and_optimizer_remain_finite(self):
        logit = torch.nn.Parameter(torch.full((2, 1, 16, 16), -60., device='cuda'))
        optimizer = torch.optim.AdamW([logit], lr=1e-4)
        prediction = logit.sigmoid()
        loss = self.loss(torch.zeros_like(prediction), prediction)
        loss.backward()
        self.assertTrue(torch.isfinite(logit.grad).all())
        optimizer.step()
        self.assertTrue(torch.isfinite(logit).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
