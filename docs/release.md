# Release checklist for maintainers

## Python packages

OpenPoints and PointNeXt use separate PyPI distributions:

- `openpoints`: the main Python library with models, datasets, layers, transforms, and training utilities.
- `pointnext_official`: PointNeXt release helpers, checkpoint downloader, and metadata.

The `pointnext` PyPI name is already occupied by another project, so do not
publish or document this release under `pip install pointnext` unless that
project is transferred to the PointNeXt maintainers.

Build and check `openpoints` first, because `pointnext_official` declares it as
a runtime dependency:

```bash
cd openpoints
python -m build --sdist --wheel --no-isolation
python -m twine check dist/*
```

Then build and check `pointnext_official` from the repository root:

```bash
cd ..
python -m build --sdist --wheel --no-isolation
python -m twine check dist/*
python -m venv /tmp/pointnext-wheel-test
/tmp/pointnext-wheel-test/bin/python -m pip install -U pip
/tmp/pointnext-wheel-test/bin/python -m pip install openpoints/dist/openpoints-*.whl
/tmp/pointnext-wheel-test/bin/python -m pip install dist/pointnext_official-*.whl
/tmp/pointnext-wheel-test/bin/python - <<'PY'
import pointnext_official
from pointnext_official.checkpoints import KNOWN_CHECKPOINTS
print(pointnext_official.__version__)
print(sorted(KNOWN_CHECKPOINTS))
PY
```

Upload requires a maintainer-owned PyPI token:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload openpoints/dist/*
python -m twine upload dist/*
```

## Checkpoints

Stage real checkpoint files and matching configs into the Hugging Face layout described in `docs/checkpoints.md`, generate `metadata/checksums.sha256`, then upload with `hf upload-large-folder`.

## GitHub release

GitHub Releases should contain source/wheel links, release notes, PyPI package names, Hugging Face checkpoint links, checksum manifest location, and the Colab/Kaggle free-GPU smoke path. Large checkpoint files should live on Hugging Face Hub rather than directly in the GitHub repository.

## Issue closing policy

Only close issues that are directly addressed by merged code/docs or a published release. For issues that require maintainer credentials or missing artifacts, comment with the PR/release plan and leave the issue open until the public artifact exists.
