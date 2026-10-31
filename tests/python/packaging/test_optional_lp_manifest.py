"""Keep public builds independent and the private shared-source build aligned."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[3]
PRIVATE = ROOT / "integrations" / "amthal"


def read_manifest(directory):
    with (directory / "Cargo.toml").open("rb") as source:
        return tomllib.load(source)


class OptionalLpManifestTests(unittest.TestCase):
    def test_public_features_and_lock_do_not_resolve_amthal(self):
        public = read_manifest(ROOT)
        self.assertNotIn("amthal", public["dependencies"])
        self.assertNotIn("lp-relaxation", public["features"])
        self.assertIn("integrations/amthal", public["workspace"]["exclude"])
        with (ROOT / "Cargo.lock").open("rb") as source:
            packages = tomllib.load(source)["package"]
        self.assertNotIn("amthal", {package["name"] for package in packages})

    def test_private_build_uses_the_same_versions_profiles_and_sources(self):
        public = read_manifest(ROOT)
        private = read_manifest(PRIVATE)
        self.assertFalse(private["package"]["publish"])
        self.assertEqual(public["workspace"]["package"], private["workspace"]["package"])
        self.assertEqual(public["profile"], private["profile"])
        for name, spec in public["dependencies"].items():
            self.assertEqual(spec, private["dependencies"][name], name)
        self.assertEqual(
            set(private["dependencies"]) - set(public["dependencies"]),
            {"amthal", "num-bigint", "num-rational", "num-traits"},
        )
        self.assertEqual(private["features"]["default"], ["lp-relaxation"])
        self.assertIn("dep:amthal", private["features"]["lp-relaxation"])
        self.assertEqual(private["features"]["python"], public["features"]["python"])
        self.assertEqual(
            (PRIVATE / private["lib"]["path"]).resolve(),
            (ROOT / public["lib"]["path"]).resolve(),
        )
        self.assertEqual(private["lib"]["crate-type"], public["lib"]["crate-type"])
        self.assertEqual(
            {item["name"]: (PRIVATE / item["path"]).resolve() for item in private["bin"]},
            {item["name"]: (ROOT / item["path"]).resolve() for item in public["bin"]},
        )
        tests = {item["name"]: (PRIVATE / item["path"]).resolve() for item in private["test"]}
        self.assertEqual(tests["lp_relaxation"], ROOT / "tests/integration/lp/lp_relaxation.rs")

    def test_private_frontends_link_the_private_core(self):
        public_workspace = read_manifest(ROOT)["workspace"]
        private_workspace = read_manifest(PRIVATE)["workspace"]
        for key in ("members", "default-members"):
            self.assertEqual(public_workspace[key], private_workspace[key])
        for member in public_workspace["members"]:
            if member == ".":
                continue
            public_dir = ROOT / member
            private_dir = PRIVATE / member
            public = read_manifest(public_dir)
            private = read_manifest(private_dir)
            with self.subTest(member=member):
                self.assertFalse(private["package"]["publish"])
                self.assertEqual(public["dependencies"], private["dependencies"])
                self.assertEqual(
                    (private_dir / private["dependencies"]["qayd"]["path"]).resolve(),
                    PRIVATE,
                )
                self.assertEqual(
                    {item["name"]: (private_dir / item["path"]).resolve() for item in private["bin"]},
                    {item["name"]: (public_dir / item["path"]).resolve() for item in public["bin"]},
                )
                if (public_dir / "src" / "lib.rs").exists():
                    self.assertEqual(
                        (private_dir / private["lib"]["path"]).resolve(),
                        public_dir / "src" / "lib.rs",
                    )

    def test_private_python_package_uses_the_same_module_and_sources(self):
        with (ROOT / "pyproject.toml").open("rb") as source:
            public = tomllib.load(source)
        with (PRIVATE / "pyproject.toml").open("rb") as source:
            private = tomllib.load(source)
        self.assertEqual(private["build-system"], public["build-system"])
        for key in ("name", "requires-python"):
            self.assertEqual(private["project"][key], public["project"][key])
        self.assertIn("version", private["project"]["dynamic"])
        public = public["tool"]["maturin"]
        private = private["tool"]["maturin"]
        for key in ("module-name", "features", "profile"):
            self.assertEqual(private[key], public[key])
        self.assertEqual(
            (PRIVATE / private["python-source"]).resolve(),
            (ROOT / public["python-source"]).resolve(),
        )


if __name__ == "__main__":
    unittest.main()
