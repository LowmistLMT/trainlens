from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1] / "skills" / "trainlens"
SCRIPT = SKILL / "scripts" / "render_report.py"
DEMO = SKILL / "assets" / "demo-logs" / "mmseg-cityscapes"


def snapshot(root: Path) -> dict:
    return {
        p.relative_to(root): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).digest()
                              if p.is_file() else None)
        for p in (root, *root.rglob("*"))
    }


class HtmlContents(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_data(self, data):
        self.text.append(data)


class ReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.experiments = self.work / "runs"
        shutil.copytree(DEMO, self.experiments)
        self.before = snapshot(self.experiments)
        self.addCleanup(self.assert_experiments_unchanged)

    def assert_experiments_unchanged(self):
        self.assertEqual(snapshot(self.experiments), self.before)

    def invoke(self, *args, ok=True):
        proc = subprocess.run(
            [sys.executable, "-E", "-S", str(SCRIPT), str(self.experiments), *map(str, args)],
            cwd=self.work, capture_output=True, text=True, timeout=15, check=False,
        )
        if ok:
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        else:
            self.assertNotEqual(proc.returncode, 0, proc.stdout)
        return proc.stdout

    def test_default_stdout_has_full_markdown_report_without_extra_files(self):
        output = self.invoke("--include-git", "--per-class", "--title", "Segmentation trial")
        self.assertIn("# Segmentation trial", output)
        self.assertRegex(output, r"\d{4}-\d{2}-\d{2}")
        for item in (
            "segformer_b2_aug", "78.65", "36000", "best_mIoU_iter_36000.pth",
            "Config Diff", "batch_size", "IoU.pole", "112233a", "445566b", "CUDA_OOM",
        ):
            with self.subTest(item=item):
                self.assertIn(item, output)
        self.assertNotIn("{{", output)
        self.assertEqual(list(self.work.iterdir()), [self.experiments])

    def test_explicit_markdown_and_metric_direction(self):
        output_file = self.work / "report.md"
        stdout = self.invoke("--output", output_file, "--title", "Training loss",
                             "--metric", "loss", "--mode", "max")
        receipt = json.loads(stdout)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["tool"], "render_report")
        self.assertEqual(Path(receipt["data"]["output"]).resolve(), output_file.resolve())
        markdown = output_file.read_text(encoding="utf-8")
        self.assertIn("# Training loss", markdown)
        self.assertIn("loss", markdown)
        self.assertIn("0.612", markdown)
        self.assertNotIn("IoU.pole", markdown)

    def test_standalone_html_escapes_custom_title(self):
        output_file = self.work / "report.html"
        title = 'Segmentation <ablation> & precision'
        self.invoke("--output", output_file, "--title", title, "--per-class")
        contents = HtmlContents()
        contents.feed(output_file.read_text(encoding="utf-8"))
        contents.close()
        self.assertIn("html", contents.tags)
        self.assertIn("style", contents.tags)
        self.assertIn("table", contents.tags)
        self.assertNotIn("script", contents.tags)
        self.assertNotIn("ablation", contents.tags)
        plain = " ".join(contents.text)
        for item in (title, "78.65", "best_mIoU_iter_36000.pth", "IoU.pole", "CUDA_OOM"):
            with self.subTest(item=item):
                self.assertIn(item, plain)

    def test_custom_template_expands_sections(self):
        builtin = (SKILL / "assets" / "report-template.md").read_text(encoding="utf-8")
        for field in ("title", "generated_at", "overview", "ranking", "config_diff",
                      "per_class", "git_provenance", "diagnostics"):
            self.assertIn("{{" + field + "}}", builtin)
        template = self.work / "custom.md"
        template.write_text(
            "# {{title}}\n\nCustom notebook {{generated_at}}\n\n"
            "{{overview}}\n\n{{ranking}}\n\n{{config_diff}}\n\n"
            "{{per_class}}\n\n{{git_provenance}}\n\n{{diagnostics}}\n",
            encoding="utf-8",
        )
        output = self.invoke("--template", template, "--title", "Custom result",
                             "--per-class", "--include-git")
        for item in ("Custom notebook", "# Custom result", "IoU.road", "445566b", "CUDA_OOM",
                     "best_mIoU_iter_36000.pth", "data_aug"):
            with self.subTest(item=item):
                self.assertIn(item, output)
        self.assertNotIn("{{", output)

    def test_does_not_write_inside_an_experiment_or_via_symlink(self):
        run_dir = self.experiments / "segformer_b0_base"
        alias = self.work / "run-link"
        alias.symlink_to(run_dir, target_is_directory=True)
        alias_file = self.work / "alias.md"
        alias_file.symlink_to(run_dir / "config.json")
        for destination in (self.experiments / "report.md", run_dir / "report.html",
                            alias / "report.md", alias_file):
            with self.subTest(destination=destination):
                self.invoke("--output", destination, ok=False)

    @unittest.skipUnless(shutil.which("git"), "Git required for local provenance")
    def test_optional_git_section_reports_head_and_dirty_worktree(self):
        repo = self.work / "code"
        repo.mkdir()
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

        def git(*args):
            result = subprocess.run(
                ["git", "-c", "user.name=TrainLens", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                cwd=repo, env=env, check=True, text=True, capture_output=True,
            )
            return result.stdout.strip()

        git("init", "-q")
        (repo / "train.txt").write_text("baseline", encoding="utf-8")
        git("add", "train.txt")
        git("commit", "-qm", "baseline")
        head = git("rev-parse", "--short", "HEAD")
        (repo / "train.txt").write_text("changed", encoding="utf-8")
        before = snapshot(repo)
        output = self.invoke("--include-git", "--repo", repo)
        self.assertIn(head, output)
        self.assertIn("dirty", output.lower())
        self.assertIn("445566b", output)
        self.assertEqual(snapshot(repo), before)
        plain = self.invoke()
        self.assertNotIn("git provenance", plain.lower())
        self.assertNotIn("445566b", plain)
