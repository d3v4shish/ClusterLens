# Model provenance and redistribution status

Audit date: 2026-09-25. This file covers models advertised by the production
UI. It records upstream evidence; it is not legal advice. Model weights are not
committed or bundled by default.

## Production clustering models

| Product model | Upstream identity | Pinned revision / acquisition | Upstream license evidence | Release policy |
| --- | --- | --- | --- | --- |
| Fast Preview | TorchVision MobileNet V3 Small | TorchVision weight enum; packaged ONNX requires its own checksum manifest | TorchVision documentation does not state a weight-specific license | Do not redistribute weights until reviewed |
| MobileCLIP S0 | `apple/mobileclip_s0_timm` | Managed snapshot pinned to `7628ba98854d84a318027e036c582df9841c556b` | [Model card](https://huggingface.co/apple/mobileclip_s0_timm) declares `apple-amlr` | Runtime download only until AMLR terms are approved |
| DINO | `timm/vit_small_patch16_224.dino` | Managed snapshot pinned to `10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4` | [Model card](https://huggingface.co/timm/vit_small_patch16_224.dino) declares Apache-2.0 | Runtime download; packaging still requires artifact review |
| DINOv2 Base | `timm/vit_base_patch14_dinov2.lvd142m` | Managed snapshot pinned to `4685c99dabffe5affac90bd99dbffd25801ae58d` | [Model card](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m) declares Apache-2.0 | Runtime download; packaging still requires artifact review |
| CLIP | `openai/clip-vit-base-patch32` | Managed snapshot pinned to `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | [Model card](https://huggingface.co/openai/clip-vit-base-patch32) has no license identifier and describes deployment limitations | Runtime download only; no redistribution approval |
| OpenCLIP | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | Managed snapshot pinned to `1a25a446712ba5ee05982a381eed697ef9b435cf` | [Model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) declares MIT and documents dataset/use caveats | Runtime download; packaging still requires review |
| SigLIP | `google/siglip-base-patch16-224` | Managed snapshot pinned to `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | [Model card](https://huggingface.co/google/siglip-base-patch16-224) declares Apache-2.0 | Runtime download; packaging still requires review |
| ResNet | TorchVision ResNet-101 | TorchVision weight enum; packaged ONNX requires its own checksum manifest | TorchVision documentation does not state a weight-specific license | Do not redistribute weights until reviewed |

The managed downloader and every corresponding online Transformers or timm
load consume the same immutable revision. Cache-only mode deliberately resolves
the locally recovered `refs/main` snapshot so an existing verified offline
cache remains usable. This prevents online acquisition from silently following
a changed upstream `main` while retaining recovery for an already verified
local snapshot.

## Human face models

The 14 metadata records under `face_model_assets/human` are the authoritative
catalog shown in Settings. They contain source links and explicit policy text.
No weights are stored in that directory.

- SCRFD and ArcFace/MobileFaceNet records are governed by the
  [InsightFace model policy](https://github.com/deepinsight/insightface): code is
  MIT, while its distributed pretrained models are non-commercial-research-only
  unless separate permission is obtained. They must not be bundled in a
  commercial release without that permission.
- YOLO5Face points to its [GPL-3.0 repository](https://github.com/deepcam-cn/yolov5-face),
  but pretrained-weight and training-data redistribution still need review.
- AdaFace points to its [MIT repository](https://github.com/mk-minchul/AdaFace),
  but pretrained-weight and training-data redistribution still need review.
- YuNet and SFace point to [OpenCV Zoo](https://github.com/opencv/opencv_zoo).
  The catalog retains the per-model policy recorded by the source metadata;
  release review must confirm the exact downloaded artifact before bundling.

## Release requirements

For every model actually shipped or qualified, the release candidate must add
an approved manifest entry containing the exact revision, every file SHA-256,
size, source URL, license/approval record, format, precision, execution target
and redistribution decision. Runtime-download-only models remain excluded from
the package. An upstream repository license is not automatically treated as a
license for separately published pretrained weights or their training data.
