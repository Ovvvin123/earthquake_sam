"""Run with: python -m unittest test_aio2_ema -v (no SAM2 dependency)."""
import ast
from contextlib import redirect_stdout
import io
from copy import deepcopy
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import nn

from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint


ROOT = Path(__file__).resolve().parent


class TinyModel(nn.Module):
    """Includes conv_sam-style aliases, dropout, frozen weights and buffers."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(7, 3, 1), nn.Dropout(0.2))
        self.backbone = nn.Conv2d(3, 3, 1).requires_grad_(False)
        self.not_sam = nn.ModuleList([self.head])
        self.in_sam = nn.ModuleList([self.backbone])
        self.register_buffer('running', torch.tensor([2.0]))
        self.register_buffer('counter', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        return self.backbone(self.head(x)).sigmoid()

    def outputImage(self, x):
        return x[:, :3]


def _function_from_file(filename, name, scope):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / filename), 'exec'), scope)
    return scope[name]


def _training_code():
    tree = ast.parse((ROOT / 'trainV10.py').read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "args.job == 'train'")
    return compile(ast.Module(body=node.body, type_ignores=[]), 'trainV10.py', 'exec')


class EMATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        np.random.seed(17)
        random.seed(17)

    def test_independent_identical_eval_only_teacher(self):
        student = TinyModel()
        teacher = EMATeacher(student)
        self.assertTrue(student.training)
        self.assertIs(teacher.model.head, teacher.model.not_sam[0])
        self.assertIsNot(student.head, teacher.model.head)
        for a, b in zip(student.parameters(), teacher.model.parameters()):
            self.assertTrue(torch.equal(a, b))
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
            self.assertFalse(b.requires_grad)
            self.assertIsNone(b.grad)
        x = torch.randn(2, 7, 4, 4, requires_grad=True)
        p1, p2 = teacher.predict(x), teacher.predict(x)
        self.assertFalse(p1.requires_grad)
        self.assertTrue(torch.equal(p1, p2))
        self.assertTrue(all(not m.training for m in teacher.model.modules()))

    def test_update_once_per_aliased_parameter_and_buffer_policy(self):
        student = TinyModel()
        teacher = EMATeacher(student, decay=0.75)
        old = {k: p.clone() for k, p in teacher.model.named_parameters()}
        with torch.no_grad():
            for p in student.parameters():
                p.add_(4.0)
            student.running.fill_(6.0)
            student.counter.fill_(9)
        teacher.update(student)
        for name, p in teacher.model.named_parameters():
            source = dict(student.named_parameters())[name]
            expected = old[name] + 1.0 if source.requires_grad else source
            torch.testing.assert_close(p, expected)
        torch.testing.assert_close(teacher.model.running, torch.tensor([3.0]))
        self.assertEqual(teacher.model.counter.item(), 9)
        self.assertEqual(teacher.num_updates, 1)

    def test_invalid_decay(self):
        for decay in [-0.1, 1.0, float('nan')]:
            with self.assertRaises(ValueError):
                EMATeacher(TinyModel(), decay)

    def test_checkpoint_restores_rng_gradients_and_teacher(self):
        student = TinyModel()
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
        teacher = EMATeacher(student, 0.8)
        student(torch.randn(2, 7, 4, 4)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        teacher.update(student)
        student(torch.randn(2, 7, 4, 4)).sum().backward()
        gradients = {k: p.grad.clone() for k, p in student.named_parameters()
                     if p.grad is not None}
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            save_ema_checkpoint(path, student, optimizer, teacher, 3, {'epochs': 3})
            expected_rng = (random.random(), np.random.rand(), torch.rand(4))
            new_student = TinyModel()
            new_optimizer = torch.optim.AdamW(new_student.parameters(), lr=9.0)
            new_teacher = EMATeacher(new_student)
            epoch = load_ema_checkpoint(path, new_student, new_optimizer,
                                        new_teacher, {'epochs': 6})
            self.assertEqual(epoch, 3)
            self.assertEqual(new_teacher.decay, 0.8)
            self.assertEqual(new_teacher.num_updates, 1)
            self.assertEqual(new_optimizer.param_groups[0]['lr'], 0.01)
            self.assertEqual(random.random(), expected_rng[0])
            self.assertEqual(np.random.rand(), expected_rng[1])
            self.assertTrue(torch.equal(torch.rand(4), expected_rng[2]))
            for name, p in new_student.named_parameters():
                if name in gradients:
                    self.assertTrue(torch.equal(p.grad, gradients[name]))
            for key, tensor in teacher.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, new_teacher.model.state_dict()[key]))
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_ema_checkpoint(path, new_student, new_optimizer,
                                    new_teacher, {'epochs': 6, 'pu': 0.4})

    def _run_training(self, initial, directory, epochs, enabled=True, resume='', preview_data=None, arg_overrides=None, batches_override=None):
        directory = str(directory) + '/'
        Path(directory).mkdir(parents=True, exist_ok=True)
        student = deepcopy(initial)
        optimizer = torch.optim.AdamW([
            {'params': student.not_sam.parameters()},
            {'params': student.in_sam.parameters()},
        ], lr=0.01)
        args = SimpleNamespace(
            lossFunc='FocalTversky', epochs=epochs, emaTeacher=enabled,
            emaDecay=0.8, emaResume=resume, resDir=directory,
            pu=0.0, puWarmup=20, puRamp=20, lr0=0.01,
            modelType='mars_sam', k0Only=True, swMode='single', isOther=False,
            isalign=False, emaPreview=preview_data is not None, previewCount=2,
            previewEvery=1, previewThreshold=.7, previewBuffer=0, previewMinPixels=1,
        )
        if arg_overrides:
            vars(args).update(arg_overrides)
        # Six batches leave a pending gradient in the legacy schedule (j=0,4).
        image = torch.linspace(0.1, 1.0, 7 * 4 * 4).reshape(1, 7, 4, 4)
        target = torch.full((1, 3, 4, 4), 0.3)
        target[:, :, :2] = 0.0
        batches = [(image.clone(), target.clone()) for _ in range(6)]
        if batches_override is not None:
            batches = batches_override
        loss = _function_from_file('torchUnet.py', 'FocalTversky', {'torch': torch})
        scope = dict(
            args=args, torch=torch, np=np, os=__import__('os'),
            net=student, opt=optimizer, resDir=directory,
            modelFile=directory + 'model.pth', plotDir=directory + 'plot/',
            device='cpu', BatchSize=1, step_per_batches=4, d0=0.02, dw=0.08,
            alpha=0.1, train_loader_train=batches, train_loader_test=batches[:1],
            FocalTversky=loss, write_config=lambda args: None,
            count_parameters=lambda net: (sum(p.numel() for p in net.parameters()),
                                         sum(p.numel() for p in net.parameters() if p.requires_grad)),
            plt=MagicMock(),
        )
        if preview_data is not None:
            scope.update(Generator_new=preview_data['dataset_type'],
                         trainFile=preview_data['filename'],
                         dataSetTrain=SimpleNamespace(list=preview_data['keys']), n=4, m=4)
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        with redirect_stdout(io.StringIO()):
            exec(_training_code(), scope)
        return student, scope['teacher']

    def test_actual_training_loop_ema_does_not_change_student(self):
        initial = TinyModel()
        with TemporaryDirectory() as directory:
            baseline, no_teacher = self._run_training(initial, Path(directory) / 'baseline', 2, False)
            trained, teacher = self._run_training(initial, Path(directory) / 'ema', 2)
            self.assertIsNone(no_teacher)
            self.assertEqual(teacher.num_updates, 4)
            for key, tensor in baseline.state_dict().items():
                self.assertTrue(torch.equal(tensor, trained.state_dict()[key]), key)
            weights = torch.load(Path(directory) / 'ema/teacher/model.pth', weights_only=True)
            for key, tensor in teacher.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, weights[key]), key)
            self.assertFalse((Path(directory) / 'baseline/teacher').exists())

    def test_actual_training_loop_resume_matches_uninterrupted(self):
        initial = TinyModel()
        with TemporaryDirectory() as directory:
            whole, whole_teacher = self._run_training(initial, Path(directory) / 'whole', 3)
            self._run_training(initial, Path(directory) / 'split', 1)
            resumed, resumed_teacher = self._run_training(
                initial, Path(directory) / 'split', 3,
                resume=str(Path(directory) / 'split/ema_checkpoint.pth'))
            self.assertEqual(resumed_teacher.num_updates, whole_teacher.num_updates)
            for key, tensor in whole.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.state_dict()[key]), key)
            for key, tensor in whole_teacher.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed_teacher.model.state_dict()[key]), key)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_prediction_update_and_restore(self):
        student = TinyModel().cuda()
        teacher = EMATeacher(student)
        optimizer = torch.optim.AdamW(student.parameters())
        x = torch.randn(2, 7, 4, 4, device='cuda')
        student(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        teacher.update(student)
        self.assertTrue(teacher.predict(x).is_cuda)
        self.assertFalse(teacher.predict(x).requires_grad)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pth'
            save_ema_checkpoint(path, student, optimizer, teacher, 1, {})
            new_student = TinyModel().cuda()
            new_teacher = EMATeacher(new_student)
            new_optimizer = torch.optim.AdamW(new_student.parameters())
            load_ema_checkpoint(path, new_student, new_optimizer, new_teacher, {})
            torch.testing.assert_close(teacher.predict(x), new_teacher.predict(x))
            # Exercise the restored optimizer's device placement.
            new_student(x).sum().backward()
            new_optimizer.step()
            new_teacher.update(new_student)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
