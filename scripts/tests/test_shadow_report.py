import json, pathlib, subprocess, sys, tempfile, unittest

SAMPLE = {
    "session": {"sessionID": "s1", "transcriptPath": "/t.md", "sessionType": "callCapture",
                "primaryModel": "Parakeet-TDT v3", "graniteModel": "granite-q8"},
    "incomplete": False,
    "segments": [
        {"startTime": 0.0, "speaker": "Speaker 2", "durationSec": 2.0,
         "primaryText": "the quarterly numbers look grim", "graniteText": "the quarterly numbers look green",
         "graniteError": None, "graniteLatencySec": 0.4},
        {"startTime": 3.0, "speaker": "Speaker 2", "durationSec": 1.5,
         "primaryText": "same words", "graniteText": "same words",
         "graniteError": None, "graniteLatencySec": 0.2},
    ],
    "totals": {"segmentCount": 2, "erroredCount": 0, "audioSeconds": 3.5,
               "shadowWallClockSec": 0.6, "rtf": 0.171},
}

SCRIPT = pathlib.Path(__file__).parent.parent / "granite-shadow-report.py"


def run_report(d: pathlib.Path, out: pathlib.Path):
    subprocess.run([sys.executable, str(SCRIPT), str(d), "-o", str(out)], check=True)


class ReportTest(unittest.TestCase):
    def test_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "s1.comparison.json").write_text(json.dumps(SAMPLE))
            out = d / "report.html"
            run_report(d, out)
            html = out.read_text()
            self.assertIn("grim", html)            # disagreement segment present
            self.assertIn("green", html)
            self.assertIn("0.171", html)           # RTF surfaced
            # disagreeing segment sorted before identical one
            self.assertLess(html.index("grim"), html.index("same words"))

    def test_error_segment_shows_badge_and_empty_granite_text(self):
        sample = json.loads(json.dumps(SAMPLE))
        sample["session"]["sessionID"] = "s2"
        sample["segments"] = [
            {"startTime": 0.0, "speaker": "Speaker 1", "durationSec": 1.0,
             "primaryText": "hello world", "graniteText": "",
             "graniteError": "requestFailed", "graniteLatencySec": 0.0},
        ]
        sample["totals"]["segmentCount"] = 1
        sample["totals"]["erroredCount"] = 1
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "s2.comparison.json").write_text(json.dumps(sample))
            out = d / "report.html"
            run_report(d, out)
            html_text = out.read_text()
            self.assertIn("hello world", html_text)
            # error surfaced somewhere near the granite cell
            self.assertIn("requestFailed", html_text)
            self.assertIn("error", html_text.lower())

    def test_malformed_file_skipped_with_warning_and_counted_in_header(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "good.comparison.json").write_text(json.dumps(SAMPLE))
            (d / "bad.comparison.json").write_text("{not valid json")
            out = d / "report.html"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(d), "-o", str(out)],
                capture_output=True, text=True, check=True)
            self.assertIn("bad.comparison.json", result.stderr)
            html_text = out.read_text()
            self.assertIn("1 unreadable", html_text)
            # the good session's content still rendered despite the sibling
            # malformed file
            self.assertIn("grim", html_text)

    def test_incomplete_session_marked(self):
        sample = json.loads(json.dumps(SAMPLE))
        sample["session"]["sessionID"] = "s3"
        sample["incomplete"] = True
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "s3.comparison.json").write_text(json.dumps(sample))
            out = d / "report.html"
            run_report(d, out)
            html_text = out.read_text()
            self.assertIn("INCOMPLETE", html_text)

    def test_html_escaping_of_transcript_text(self):
        sample = json.loads(json.dumps(SAMPLE))
        sample["session"]["sessionID"] = "s4"
        sample["segments"] = [
            {"startTime": 0.0, "speaker": "Speaker 1", "durationSec": 1.0,
             "primaryText": "click <script>alert(1)</script> now",
             "graniteText": "click <script>alert(1)</script> now",
             "graniteError": None, "graniteLatencySec": 0.1},
        ]
        sample["totals"]["segmentCount"] = 1
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "s4.comparison.json").write_text(json.dumps(sample))
            out = d / "report.html"
            run_report(d, out)
            html_text = out.read_text()
            self.assertNotIn("<script>alert(1)</script>", html_text)
            self.assertIn("&lt;script&gt;", html_text)


if __name__ == "__main__":
    unittest.main()
