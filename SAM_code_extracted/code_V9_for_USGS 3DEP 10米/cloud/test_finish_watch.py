"""Offline watcher contracts. Never runs shutdown, a trainer, or a network call."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import finish_watch as watch
sys.path.insert(0, str(watch.ROOT))


class FinishTests(unittest.TestCase):
    def launch(self, status='running'):
        return dict(status=status, resolved_config=dict(epochs=500, emaStartEpoch=250, emaTeacher=True))

    def artifacts(self, run):
        for name in ('ema_checkpoint.pth', 'model.pth', 'resolved_config.json', 'console.log'):
            (run/name).write_bytes(b'offline test fixture')

    def test_final_epoch_alone_never_triggers_completion(self):
        with TemporaryDirectory() as tmp:
            self.assertFalse(watch.completion_ready(Path(tmp), self.launch(), [{'epoch':500}]))
            self.assertFalse(watch.completion_ready(Path(tmp), self.launch('failed_or_interrupted'), [{'epoch':500}]))

    def test_complete_requires_matching_final_epoch_and_nonempty_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'history'):
                watch.completion_ready(root, self.launch('complete'), [{'epoch':499}])
            with self.assertRaisesRegex(RuntimeError, 'artifact'):
                watch.completion_ready(root, self.launch('complete'), [{'epoch':500}])
            self.artifacts(root)
            self.assertTrue(watch.completion_ready(root, self.launch('complete'), [{'epoch':500}]))
            (root/'model.pth').write_bytes(b'')
            with self.assertRaisesRegex(RuntimeError, 'artifact'):
                watch.completion_ready(root, self.launch('complete'), [{'epoch':500}])

    def test_partial_history_tail_tolerated_but_internal_corruption_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'history.jsonl'
            path.write_text('{"epoch": 1}\n{"epoch":')
            self.assertEqual(watch.history_rows(path), [{'epoch':1}])
            path.write_text('broken\n{"epoch": 2}\n')
            with self.assertRaises(json.JSONDecodeError): watch.history_rows(path)

    def test_eta_changes_after_actual_late_phase_measurements(self):
        rows = [dict(epoch=e, epoch_seconds=60) for e in range(1, 6)]
        early = watch.progress(self.launch(), rows)
        self.assertGreater(early['estimated_remaining_hours'], (500-5)/60)
        rows += [dict(epoch=e, epoch_seconds=120) for e in (250, 251, 252)]
        late = watch.progress(self.launch(), rows)
        self.assertEqual(late['estimated_remaining_hours'], round((500-252)*120/3600, 2))
        self.assertIsNone(watch.progress(self.launch(), [])['estimated_remaining_hours'])

    def test_manifest_preserves_results_and_excludes_live_monitor_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.artifacts(root)
            destination = root/'auto_finish'/'fixture'
            destination.mkdir(parents=True)
            (destination/'settings.txt').write_text('snapshot')
            watch.finalize(root, destination, self.launch('complete'), [{'epoch':500}])
            manifest = json.loads((destination/'results_manifest.json').read_text())
            self.assertEqual(len(manifest), 4)
            self.assertEqual(manifest['model.pth']['sha256'], watch.sha256(root/'model.pth'))
            self.assertEqual((root/'model.pth').read_bytes(), b'offline test fixture')

    def test_source_drift_refuses_snapshot(self):
        with TemporaryDirectory() as tmp, patch('training_sources.training_source_snapshot', return_value={}):
            launch = self.launch()
            launch['preflight'] = {'source_sha256': {'trainV10.py':'changed'}}
            with self.assertRaisesRegex(RuntimeError, 'fingerprint'):
                watch.capture_setup(Path(tmp), Path(tmp), launch)


if __name__ == '__main__':
    unittest.main()
