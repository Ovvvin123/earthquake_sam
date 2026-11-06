"""Tests for read-only previews; run with python -m unittest test_aio2_preview -v."""
import ast
from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.utils.data as data_utils

import test_aio2_ema as ema_tests
from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint
from aio2_preview import FixedPreview, BatchDiagnostics, candidate_masks, probability_stats


def dataset_class():
    tree = ast.parse((ema_tests.ROOT / 'tool.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Generator_new')
    scope = {'np': np, 'File': h5py.File, 'zoom': zoom, 'data_utils': data_utils}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'tool.py', 'exec'), scope)
    return scope['Generator_new']


def make_data(path):
    with h5py.File(path, 'w') as h5:
        for index in range(4):
            group = h5.create_group(f'tile_{index}')
            image = np.arange(64, dtype=np.float32).reshape(8, 8)
            group['CTX'] = image
            group['image'] = image - 2000
            group['TW'] = np.stack([image, image, image])
            distance = np.full((8, 8), .3, dtype=np.float32)
            if index % 2 == 0:
                distance[2:6, 3] = 0
            group['k0'] = distance
            group['k1'] = distance
            group['k2'] = np.full((8, 8), 999, dtype=np.float32)
    return [f'tile_{i}' for i in range(4)]


class PreviewTests(unittest.TestCase):
    def test_center_crop_is_fixed_and_does_not_consume_numpy_rng(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'data.h5')
            keys = make_data(path)
            cls = dataset_class()
            dataset = cls(path, keys, n=4, m=4, training=False, fixed_crop=True)
            np.random.seed(33)
            state = np.random.get_state()
            image1, label1 = dataset[0]
            actual = np.random.rand()
            np.random.set_state(state)
            expected = np.random.rand()
            self.assertEqual(actual, expected)
            image2, label2 = dataset[0]
            np.testing.assert_array_equal(image1, image2)
            np.testing.assert_array_equal(label1, label2)
            self.assertEqual(int((label1[0] < .02).sum()), 4)
            self.assertEqual(cls(path, keys, n=8, m=8, fixed_crop=True)[0][0].shape, (7, 8, 8))
            with self.assertRaises(ValueError):
                cls(path, keys, fixed_crop=True, training=True)

    def test_candidates_exclude_labels_buffer_invalid_and_small_components(self):
        p = np.zeros((9, 9))
        truth = np.zeros_like(p, dtype=bool)
        truth[:, 1] = True
        valid = np.ones_like(truth)
        p[1:7, 1:3] = .9
        p[2:7, 5] = .9
        p[8, 8] = .9
        p[0, 5] = .9
        valid[0, 5] = False
        raw, selected, raw_count, selected_count = candidate_masks(p, truth, valid, .7, 1, 3)
        self.assertEqual(raw_count, 2)
        self.assertEqual(selected_count, 1)
        self.assertEqual(int(selected.sum()), 5)
        self.assertFalse((selected & truth).any())
        self.assertFalse(selected[0, 5])
        self.assertFalse(selected[8, 8])
        _, no_buffer, _, _ = candidate_masks(p, truth, valid, .7, 0, 1)
        self.assertTrue(no_buffer[2, 2])
        self.assertFalse(no_buffer[2, 1])

    def test_invalid_only_statistics_are_json_safe(self):
        result = probability_stats(np.array([np.nan]), np.array([False]), .7)
        self.assertIsNone(result['mean'])
        self.assertEqual(sum(result['histogram']), 0)
        json.dumps(result, allow_nan=False)

    def test_fixed_preview_outputs_and_preserves_weights_modes_and_rng(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'data.h5')
            keys = make_data(path)
            preview = FixedPreview(dataset_class(), path, keys, Path(directory) / 'preview',
                                   4, 4, .02, count=2, min_pixels=1)
            # Order of the incoming training key list cannot change selection.
            same = FixedPreview(dataset_class(), path, keys[::-1], Path(directory) / 'preview',
                                4, 4, .02, count=2, min_pixels=1)
            self.assertEqual([s[0] for s in preview.samples], [s[0] for s in same.samples])
            student = ema_tests.TinyModel()
            student.train()
            student.in_sam.eval()
            teacher = EMATeacher(student)
            state = deepcopy(student.state_dict())
            modes = [module.training for module in student.modules()]
            rng = torch.get_rng_state().clone()
            result = preview.write(student, teacher, 4)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(modes, [module.training for module in student.modules()])
            self.assertEqual(teacher.num_updates, 0)
            for key, tensor in student.state_dict().items():
                self.assertTrue(torch.equal(tensor, state[key]))
            report = json.loads((result / 'metrics.json').read_text())
            self.assertTrue(report['preview_only'])
            self.assertEqual(len(report['samples']), 2)
            for index in range(2):
                self.assertGreater((result / f'sample_{index:02d}.png').stat().st_size, 1000)
                with np.load(result / f'sample_{index:02d}.npz') as arrays:
                    np.testing.assert_array_equal(arrays['original_label'], preview.samples[index][2])
            with self.assertRaisesRegex(ValueError, 'samples changed'):
                FixedPreview(dataset_class(), path, keys, Path(directory) / 'preview',
                             4, 4, .02, count=3)

    def test_actual_warmup_preview_without_teacher(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'data.h5')
            keys = make_data(path)
            runner = ema_tests.EMATests()
            student, teacher = runner._run_training(
                ema_tests.TinyModel(), Path(directory) / 'warmup', 1,
                preview_data=dict(dataset_type=dataset_class(), filename=path, keys=keys),
                arg_overrides={'emaStartEpoch': 3})
            self.assertIsNone(teacher)
            output = Path(directory) / 'warmup/preview/epoch_0001'
            report = json.loads((output / 'metrics.json').read_text())
            self.assertFalse(report['teacher_active'])
            self.assertEqual(report['ema_updates'], 0)
            for index, record in enumerate(report['samples']):
                self.assertIsNone(record['teacher'])
                self.assertEqual(record['filtered_candidate_pixels'], 0)
                self.assertGreater((output / f'sample_{index:02d}.png').stat().st_size, 1000)
                with np.load(output / f'sample_{index:02d}.npz') as arrays:
                    self.assertTrue(np.isnan(arrays['teacher']).all())
                    self.assertTrue(np.isfinite(arrays['student']).all())

    def test_batch_diagnostics_distinguish_empty_and_positive_near_one_loss(self):
        diagnostic = BatchDiagnostics()
        image = torch.ones(1, 7, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        prediction = torch.full((1, 3, 4, 4), .8, requires_grad=True)
        diagnostic.observe(0, target, prediction, image, 1.0)
        target[:, :, 0, 0] = 1
        diagnostic.observe(1, target, prediction, image, 1.0)
        self.assertIsNone(prediction.grad)
        with TemporaryDirectory() as directory:
            diagnostic.write(directory, 5)
            output = Path(directory) / 'epoch_0005'
            report = json.loads((output / 'train_summary.json').read_text())
            for group in ('empty_label', 'with_label'):
                self.assertEqual(report['groups_by_original_label'][group]['near_one_loss_batches'], 1)
            with (output / 'train_batches.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_old_checkpoint_allows_new_diagnostics_but_not_loss_changes(self):
        student = ema_tests.TinyModel()
        opt = torch.optim.AdamW(student.parameters())
        teacher = EMATeacher(student)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pth'
            save_ema_checkpoint(path, student, opt, teacher, 4, {'pu': 0.0})
            config = {'pu': 0.0, 'emaPreview': True, 'previewCount': 6,
                      'previewEvery': 1, 'previewThreshold': .7, 'previewBuffer': 2, 'previewMinPixels': 8}
            self.assertEqual(load_ema_checkpoint(path, student, opt, teacher, config), 4)
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_ema_checkpoint(path, student, opt, teacher, dict(config, pu=.4))

    def test_actual_training_loop_with_preview_keeps_student_identical(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'data.h5')
            keys = make_data(path)
            runner = ema_tests.EMATests()
            runner.setUp()
            initial = ema_tests.TinyModel()
            baseline, _ = runner._run_training(initial, Path(directory) / 'baseline', 1)
            preview_data = {'filename': path, 'keys': keys, 'dataset_type': dataset_class()}
            enabled, teacher = runner._run_training(initial, Path(directory) / 'enabled', 1,
                                                    preview_data=preview_data)
            for key, tensor in baseline.state_dict().items():
                self.assertTrue(torch.equal(tensor, enabled.state_dict()[key]), key)
            report = Path(directory) / 'enabled/preview/epoch_0001/metrics.json'
            self.assertTrue(report.exists())
            self.assertEqual(teacher.num_updates, 2)
            self.assertTrue((report.parent / 'train_batches.csv').exists())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
