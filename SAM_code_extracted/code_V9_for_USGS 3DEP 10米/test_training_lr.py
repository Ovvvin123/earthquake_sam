"""CPU schedule and real checkpoint-resume contracts; never starts training."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from training_lr import learning_rate_factors, schedule_config

NEW = dict(lrSchedule='warmup_hold_cosine', lrWarmupEpochs=10,
           lrHoldEnd=100, lrDecayEnd=500, lrMinFactor=.125)


class ScheduleTests(unittest.TestCase):
    def test_legacy_exactly_preserved(self):
        for i in (0, 1, 2, 3, 4, 7, 8, 249, 499, 599):
            for j in (0, 1, 173):
                head = 1. if i < 2 else .5 if i < 4 else .25 if i < 8 else .125
                sam = 1. if i < 2 else .5 if i < 4 else .25
                if i == 0:
                    head = sam = 1 - np.exp(-j / 174 * 3)
                self.assertEqual(learning_rate_factors({}, i, j, 174), (head, sam))

    def test_warmup_hold_decay_and_extension(self):
        warm = [learning_rate_factors(NEW, i, j, 174)[0] for i in range(10) for j in range(174)]
        self.assertGreater(warm[0], 0)
        self.assertEqual(warm[-1], 1)
        self.assertTrue(all(a < b for a, b in zip(warm, warm[1:])))
        self.assertEqual(learning_rate_factors(NEW, 10, 0, 174), (1., 1.))
        self.assertEqual(learning_rate_factors(NEW, 99, 173, 174), (1., 1.))
        decay = [learning_rate_factors(NEW, i, 0, 174)[0] for i in range(100, 500)]
        self.assertTrue(all(a > b for a, b in zip(decay, decay[1:])))
        self.assertEqual(decay[-1], .125)
        self.assertEqual(learning_rate_factors(dict(NEW, epochs=1000), 999, 0, 174), (.125, .125))
        self.assertAlmostEqual(learning_rate_factors(NEW, 299, 0, 174)[0], .5625)

    def test_reject_invalid_configuration(self):
        for changes in (dict(lrWarmupEpochs=0), dict(lrHoldEnd=9), dict(lrDecayEnd=100),
                        dict(lrMinFactor=float('nan')), dict(lrMinFactor=0),
                        dict(lrMinFactor=1.1), dict(lrSchedule='typo')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                schedule_config(dict(NEW, **changes))

    def test_production_profile_parses_without_running_trainer(self):
        from run_experiment import profile_arguments, command, validate_parser
        profile = Path(__file__).parent / 'experiments/california_v9_30m_1024_8gb.json'
        args = validate_parser(command(profile_arguments(profile)))
        self.assertEqual(schedule_config(args), NEW)
        self.assertEqual((args['emaStartEpoch'], args['unimatchRampEnd']), (250, 350))
        self.assertEqual((args['lr0'], args['loraLr']), (2e-4, 1e-4))

    def test_optimizer_groups_and_both_checkpoint_types(self):
        import torch
        from sam2_lora import optimizer_groups, set_learning_rates, normalized_config
        from test_sam2_lora import TinyModel, options
        from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint
        from training_checkpoint import save_supervised_checkpoint, load_supervised_checkpoint
        args = options()
        vars(args).update(NEW, lr0=2e-4, loraLr=1e-4, epochs=500, emaTeacher=True)
        model = TinyModel(vars(args))
        opt = torch.optim.AdamW(optimizer_groups(model, args))
        teacher = EMATeacher(model)
        cfg = vars(args)
        with TemporaryDirectory() as tmp:
            for ema in (False, True):
                with self.subTest(ema=ema):
                    path = Path(tmp) / ('ema.pth' if ema else 'supervised.pth')
                    set_learning_rates(opt, args, *learning_rate_factors(args, 249, 173, 174))
                    if ema:
                        save_ema_checkpoint(path, model, opt, teacher, 250, cfg)
                    else:
                        save_supervised_checkpoint(path, model, opt, 250, 0, cfg)
                    resumed = TinyModel(cfg)
                    resumed_opt = torch.optim.AdamW(optimizer_groups(resumed, args))
                    resumed_teacher = EMATeacher(resumed)
                    def load(c):
                        if ema:
                            return load_ema_checkpoint(path, resumed, resumed_opt, resumed_teacher, c)
                        return load_supervised_checkpoint(path, resumed, resumed_opt, c)[0]
                    self.assertEqual(load(dict(cfg, epochs=600)), 250)
                    factors = learning_rate_factors(args, 250, 0, 174)
                    set_learning_rates(opt, args, *factors)
                    set_learning_rates(resumed_opt, args, *factors)
                    self.assertEqual([g['lr'] for g in opt.param_groups],
                                     [g['lr'] for g in resumed_opt.param_groups])
                    self.assertAlmostEqual(resumed_opt.param_groups[0]['lr'], 2e-4*factors[0])
                    self.assertAlmostEqual(resumed_opt.param_groups[1]['lr'], 1e-4*factors[0])
                    for key, value in (('lrSchedule', 'legacy'), ('lrHoldEnd', 150),
                                       ('lrDecayEnd', 600), ('lrMinFactor', .1)):
                        with self.assertRaisesRegex(ValueError, key):
                            load(dict(cfg, **{key: value}))
            self.assertEqual(normalized_config({})['lrSchedule'], 'legacy')


if __name__ == '__main__':
    unittest.main()
