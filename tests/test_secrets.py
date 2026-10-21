
import ast
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SECRET_PREFIXES = [
    "gsk" + "_",
    "hf" + "_",
    "sk" + "-proj-",
]

SECRET_PATTERN = re.compile(
    r"(?:" +
    "|".join(re.escape(p) for p in SECRET_PREFIXES) +
    r")[A-Za-z0-9_-]{16,}"
)

SENSITIVE_ARGUMENTS = {
    "api_key",
    "secret_key",
    "access_token",
}


def source_files():
    files = [
        path for path in PROJECT_ROOT.glob("*.py")
        if not path.name.startswith("test_")
    ]

    for folder, extension in [
        ("static", "*.js"),
        ("templates", "*.html"),
    ]:
        directory = PROJECT_ROOT / folder

        if directory.exists():
            files.extend(directory.rglob(extension))

    return files


def find_hardcoded_secrets():
    problems = []

    for path in source_files():
        content = path.read_text(
            encoding="utf-8-sig",
            errors="replace"
        )

        # Look for recognizable secret-token prefixes.
        for match in SECRET_PATTERN.finditer(content):
            line_number = (
                content.count("\n", 0, match.start()) + 1
            )
            problems.append(
                f"{path.name}:{line_number} - "
                "Possible embedded secret token"
            )

        # Check Python calls with literal API credentials.
        if path.suffix == ".py":
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if (
                            keyword.arg in SENSITIVE_ARGUMENTS
                            and isinstance(
                                keyword.value, ast.Constant
                            )
                            and isinstance(
                                keyword.value.value, str
                            )
                            and len(keyword.value.value) >= 16
                        ):
                            problems.append(
                                f"{path.name}:"
                                f"{node.lineno} - "
                                "Hardcoded credential argument"
                            )

    return sorted(set(problems))


class ChatScholarSecretTests(unittest.TestCase):

    def test_no_hardcoded_secrets(self):
        problems = find_hardcoded_secrets()

        self.assertEqual(
            problems,
            [],
            "Potential secrets detected:\n"
            + "\n".join(problems)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
