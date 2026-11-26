from __future__ import annotations

import json
import re
import textwrap
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / "skills" / "trainlens"
RESOURCE = re.compile(
    r"(?<![\w-])(?:scripts/[\w.@+%~/-]+\.py\b|references/[\w.@+%~/-]+\.md\b"
    r"|assets/demo-logs\b(?:/[\w.@+%~/-]*)?)"
)


def without_comment(scalar: str) -> str:
    if scalar.startswith(('"', "'")):
        quoted = re.fullmatch(r"""("(?:\\.|[^"\\])*"|'(?:''|[^'])*')(?:\s+#.*)?""", scalar)
        if not quoted:
            raise ValueError(f"Invalid quoted manifest value: {scalar}")
        return quoted[1]
    return re.split(r"(?:^|\s)#", scalar, maxsplit=1)[0].rstrip()


def read_mapping(text: str) -> dict:
    """Read the nested mappings and scalar styles used by these two manifests."""
    fields = {}
    text = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    # A line at the current indentation starts a key; indented lines belong to it.
    for block in re.split(r"\n(?=[^\s#])", text.strip("\n")):
        lines = block.splitlines()
        key, colon, scalar = lines[0].partition(":")
        if not colon or not key.isidentifier() or key in fields:
            raise ValueError(f"Invalid or repeated manifest key: {key!r}")
        body = textwrap.dedent("\n".join(lines[1:]))
        content = any(line.strip() and not line.lstrip().startswith("#") for line in lines[1:])
        scalar = without_comment(scalar.strip())
        if scalar in (">", ">-", ">+", "|", "|-", "|+"):
            if scalar.startswith(">"):
                body = re.sub(r"(?<=\S)\n(?=\S)", " ", body)
            value = body if scalar.endswith("-") else body + "\n"
        elif not scalar:
            value = read_mapping(body) if content else None
        elif content:
            raise ValueError(f"Unexpected indented content after {key!r}")
        elif scalar.startswith("'") and scalar.endswith("'"):
            value = scalar[1:-1].replace("''", "'")
        elif scalar == "~" or scalar.lower() == "null":
            value = None
        elif scalar.lower() in ("true", "false"):
            value = scalar.lower() == "true"
        else:
            try:
                value = json.loads(scalar)
            except json.JSONDecodeError:
                # Do not mistake malformed quotes or a collection for a plain string.
                if scalar.startswith(('"', '[', '{', '!', '&', '*', '- ')) or re.search(r":(?:\s|$)", scalar):
                    raise ValueError(f"Expected a manifest scalar for {key!r}")
                value = scalar
        fields[key] = value
    return fields


class SkillManifestTests(unittest.TestCase):
    def test_skill_frontmatter_and_length(self):
        lines = (SKILL / "SKILL.md").read_text(encoding="utf-8").splitlines()
        self.assertLess(len(lines), 300)
        self.assertEqual(lines[0], "---")
        self.assertIn("---", lines[1:], "Missing frontmatter closing delimiter")
        closing = lines.index("---", 1)
        metadata = read_mapping("\n".join(lines[1:closing]))
        self.assertEqual(set(metadata), {"name", "description"})
        self.assertEqual(metadata["name"], SKILL.name)
        self.assertEqual(metadata["name"], "trainlens")
        self.assertLessEqual(len(metadata["name"]), 64)
        description = metadata["description"]
        self.assertIsInstance(description, str)
        self.assertTrue(description.strip())
        self.assertLessEqual(len(description), 1024)

    def test_openai_interface_and_invocation_policy(self):
        metadata = read_mapping((SKILL / "agents" / "openai.yaml").read_text(encoding="utf-8"))
        interface = metadata["interface"]
        self.assertIsInstance(interface, dict)
        self.assertIsInstance(metadata["policy"], dict)
        for key in ("display_name", "short_description", "default_prompt"):
            with self.subTest(key=key):
                self.assertIsInstance(interface[key], str)
                self.assertTrue(interface[key].strip())
        self.assertIsInstance(metadata["policy"]["allow_implicit_invocation"], bool)

    def test_referenced_skill_resources_exist_in_repository(self):
        content = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        resources = {match.rstrip(".") for match in RESOURCE.findall(content)}
        self.assertEqual({name.split("/")[0] for name in resources},
                         {"scripts", "references", "assets"})
        for name in sorted(resources):
            with self.subTest(reference=name):
                path = (SKILL / name).resolve()
                self.assertTrue(path.is_relative_to(REPO), f"Outside repository: {name}")
                if name.startswith(("scripts/", "references/")):
                    self.assertTrue(path.is_file(), f"Missing file: {name}")
                elif name.endswith("/") or name == "assets/demo-logs":
                    self.assertTrue(path.is_dir(), f"Missing directory: {name}")
                else:
                    self.assertTrue(path.exists(), f"Missing demo path: {name}")
