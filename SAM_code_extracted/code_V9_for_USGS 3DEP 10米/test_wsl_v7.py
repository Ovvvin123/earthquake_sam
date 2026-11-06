"""Storage and full-tile integration checks; no formal training."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from epoch_backup import due, publish_epoch, backup_epoch, sha256
from sam2_lora import normalized_config
from run_experiment import read_profile


class StorageTests(unittest.TestCase):
    def test_schedule_and_default_disabled(self):
        cfg = dict(backupDir='/backup', backupEvery=20, backupMilestones='249,350,500', epochs=500)
        for epoch in (20, 240, 249, 260, 350, 500):
            self.assertTrue(due(epoch, cfg))
        for epoch in (1, 19, 248, 250, 349):
            self.assertFalse(due(epoch, cfg))
        self.assertFalse(due(20, {}))

    def test_closed_snapshot_hashes_and_final_best_weights(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'; run.mkdir()
            (run/'ema_checkpoint.pth').write_bytes(b'closed full resume snapshot')
            (run/'resolved_config.json').write_text('{}')
            (run/'console.log').write_text('live stream is excluded')
            best = run/'best_student'; best.mkdir(); (best/'model.pth').write_bytes(b'best weights')
            cfg = dict(backupDir=str(Path(tmp)/'backup'), emaTeacher=True, epochs=500)
            out = publish_epoch(run, 500, cfg)
            manifest = json.loads((out/'backup_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertNotIn('console.log', manifest['files'])
            self.assertIn('best_student/model.pth', manifest['files'])
            for name, info in manifest['files'].items():
                self.assertEqual(sha256(out/name), info['sha256'])
            with self.assertRaises(FileExistsError):
                publish_epoch(run, 500, cfg)
            with patch('epoch_backup.sha256', return_value='bad'):
                with self.assertRaises(IOError):
                    publish_epoch(run, 20, cfg)
            self.assertFalse((Path(cfg['backupDir'])/'epoch_0020').exists())
            cfg.update(backupEvery=20)
            with patch('epoch_backup.publish_epoch', side_effect=OSError('disk full')):
                with self.assertWarns(UserWarning):
                    self.assertIsNone(backup_epoch(run, 20, cfg))
            self.assertTrue((run/'ema_checkpoint.pth').exists())

    def test_old_default_and_new_profile(self):
        self.assertFalse(normalized_config({})['fullTile'])
        root = Path(__file__).parent/'experiments'
        old = read_profile(root/'california_v8_land12km_ds6.json')
        new = read_profile(root/'california_v7_30m_1024_wsl.json')
        self.assertNotIn('fullTile', old)
        self.assertNotIn('backupDir', old)
        self.assertEqual(new['mn'], '1024')
        self.assertEqual(new['lr0'], old['lr0']*2)
        self.assertEqual(new['loraLr'], old['loraLr']*2)
        self.assertEqual((new['emaStartEpoch'], new['unimatchRampEnd'], new['epochs']), (250,350,500))


if __name__ == '__main__':
    unittest.main()
