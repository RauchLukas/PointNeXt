# FAQ and common installation/runtime questions

## I cloned the repo but `openpoints` is missing

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/guochengqian/PointNeXt.git
cd PointNeXt
git submodule update --init --recursive
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

For a pure Python import check, the OpenPoints library can also be installed with:

```bash
pip install openpoints
```

For training/evaluation, build the CUDA/C++ extensions from a source checkout.

## Does `pip install pointnext_official` include CUDA ops?

No. The PyPI packages make the Python modules importable and provide metadata/checkpoint helpers. PointNeXt training/evaluation still uses custom CUDA/C++ operators that depend on the local Python, PyTorch, CUDA, compiler, and platform ABI. Build them from source:

```bash
cd openpoints/cpp/pointnet2_batch && python setup.py install && cd ../../..
cd openpoints/cpp/pointops && python setup.py install && cd ../../..
```

`chamfer_dist` and `emd` are optional for classification/segmentation and mainly needed for reconstruction/completion tasks.

## Can I run PointNeXt on CPU only?

CPU-only import and packaging smoke tests are supported. The main PointNeXt models rely on CUDA custom ops such as ball query / grouping / pointops for practical training and evaluation, so full benchmark reproduction should be run on a CUDA GPU.

## What does `in_channels` mean?

`in_channels` is the number of per-point input feature channels consumed by the encoder.

Examples:

- ModelNet40 PointNeXt-S uses xyz only: `in_channels=3`.
- ScanObjectNN PointNeXt-S uses xyz plus an extra feature/height channel in this config: `in_channels=4`.
- Segmentation configs may use xyz plus color/height/features depending on the dataset pipeline.

A checkpoint must be evaluated with a config that matches its `in_channels`, width, number of classes, and dataset preprocessing.

## Why does ModelNet40 testing use `model.encoder_args.width=64`?

The default `cfgs/modelnet40ply2048/pointnext-s.yaml` is PointNeXt-S width 32. The released ModelNet40 model-zoo checkpoint is the C=64 variant, so testing that checkpoint requires:

```bash
model.encoder_args.width=64
```

Training the default width-32 config does not need this override.

## Headless server visualization crashes or opens no window

Visualization utilities may require an OpenGL context. On remote/headless servers, prefer offscreen rendering or run through a virtual display:

```bash
export PYVISTA_OFF_SCREEN=true
xvfb-run -s "-screen 0 1024x768x24" python examples/segmentation/vis_results.py ...
```

If visualization still segfaults, first verify the non-visual evaluation command on the same checkpoint/config, then report the OS, GPU driver, CUDA, PyTorch, PyVista, and OpenGL/Mesa versions.

## How do I save segmentation predictions as `.obj` files?

For S3DIS and related segmentation scripts, add `visualize=True` to the test command:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
  --cfg cfgs/s3dis/pointnext-s.yaml \
  mode=test \
  --pretrained_path /path/to/checkpoint.pth \
  visualize=True
```

The visualization branch is implemented in `examples/segmentation/main.py`. ScanNet test data may not contain labels, so prediction/input `.obj` files can be generated directly; ground-truth visualization requires evaluating a labeled split and using a color map whose indices match the dataset labels.

## How do I train or test all S3DIS areas?

The standard configs train/test one held-out area at a time through `dataset.common.test_area`. For six-fold S3DIS evaluation, run each fold/checkpoint and then aggregate with:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/test_s3dis_6fold.py \
  --cfg cfgs/s3dis/pointnext-xl.yaml \
  mode=test \
  --pretrained_path pretrained/s3dis/pointnext-xl
```

## Why are validation/test results different from the automatic result after training?

For S3DIS-style segmentation, validation during training is usually performed on sampled/subsampled point clouds for speed. Final test mode evaluates the full scenes with the configured voting/test pipeline. Use the standalone `mode=test --pretrained_path ...` command for the reportable number.

## How do I change the optimizer in a config?

Set the `optimizer` block in the YAML file or override it from the command line:

```yaml
optimizer:
  NAME: adamw
  weight_decay: 1.0e-4
```

Common names are defined by `openpoints/optim/optim_factory.py`.

## How do I reduce S3DIS/ScanNet test memory usage?

Full-scene segmentation testing can use substantially more memory than training batches. On smaller GPUs, reduce the number of votes, lower `voxel_max` for the test split, use a smaller model/config, disable visualization, and resume testing directly from the checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
  --cfg cfgs/s3dis/pointnext-s.yaml \
  mode=test \
  --pretrained_path /path/to/checkpoint.pth \
  num_votes=1 \
  visualize=False
```

## Where is the preprocessing code?

Dataset preprocessing lives with the dataset loaders. Common entry points include:

- S3DIS/ScanNet docs: `docs/examples/s3dis.md`, `docs/examples/scannet.md`
- S3DIS loader/cache path: `openpoints/dataset/s3dis/s3dis.py`
- SemanticKITTI/Semantic3D preprocessing helpers: `openpoints/dataset/semantic_kitti/utils/`

Some datasets also support automatic local cache generation the first time the dataset class is instantiated.

## What does `part_seg_refinement` do?

`part_seg_refinement` in `examples/shapenetpart/main.py` is a ShapeNetPart post-processing step. It restricts predictions to the valid part labels for the object category, then uses nearby points to replace invalid part predictions. Final ShapeNetPart metrics are computed on the 2048 sampled test points used by the evaluation pipeline.

## Are instance segmentation or custom detector backbones supported?

PointNeXt/OpenPoints mainly provides classification, part segmentation, and semantic segmentation examples. Instance segmentation and detector-backbone integrations, such as replacing PointRCNN backbones, are possible research extensions but are not maintained as supported example pipelines here. When adapting PointNeXt to a new detector or dataset, tune the radius/receptive-field schedule carefully; the default segmentation radii are not guaranteed to transfer.

## Where are pretrained checkpoints?

Use the model-zoo docs and checkpoint helper:

```bash
pip install pointnext_official
pointnext-download --list
```

Large checkpoint files are staged outside PyPI. If a specific checkpoint is not listed, it has not been published in the maintained release layout yet.

## `Permission denied` when running a Python file

Run Python scripts through Python, not as shell executables:

```bash
python examples/classification/main.py --cfg cfgs/modelnet40ply2048/pointnext-s.yaml
```

If a shell script fails with permission denied, either run it with `bash script.sh` or mark it executable with `chmod +x script.sh`.
