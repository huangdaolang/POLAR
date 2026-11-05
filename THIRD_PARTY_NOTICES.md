# Third-party notices

POLAR's original code is licensed under the repository-level MIT license. The
vendored backbone implementations and model weights retain their upstream
licenses and are not relicensed as MIT.

## TabICL

- Local scope: `src/polar/backbones/tabicl/model/`
- Upstream: https://github.com/soda-inria/tabicl
- Source revision: `f719c886a586ed4a29236345e319ac1ea596c478`
- License: BSD 3-Clause; see `licenses/TABICL_LICENSE.txt`
- Official regression checkpoint:
  `jingang/TabICL/tabicl-regressor-v2-20260212.ckpt`
- Checkpoint SHA-256:
  `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`

POLAR changes the import namespace, exposes candidate-conditioned hidden
representations, disables inference AMP by default, and includes an asynchronous
copy-stream lifetime fix. The surrounding loader and backbone wrapper are POLAR
code.

## TabPFN

- Local scope: `src/polar/backbones/tabpfn/model/`
- Upstream: https://github.com/PriorLabs/TabPFN
- Core source revision: `43116016c379ece0e21247197d0f75a65a30433e`
- Pre-refactor encoder/static-embedding revision:
  `827afacff93de5e8897e65c51954a0ea5c4a42d5`
- Code license: Prior Labs License v1.2; see
  `licenses/TABPFN_LICENSE.txt`

The local copy is a regression-oriented minimal extraction. POLAR changes the
import namespace, removes unused inference/classification paths, and exposes
candidate-conditioned hidden representations. Modified files are identifiable
by comparison with the pinned revisions above. The public repository starts
from a release snapshot rather than the development git history.

TabPFN-2.5 weights are governed by a separate non-commercial license; see
`licenses/TABPFN_2_5_MODEL_LICENSE.txt`, copied unchanged from
`Prior-Labs/tabpfn_2_5` at revision
`6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b`. Any
redistribution of the original or fine-tuned TabPFN-2.5 weights must include the
applicable model license and Prior Labs attribution notice. POLAR does not
relicense those weights.

## Non-vendored dependencies and datasets

PyTorch, NumPy, Hydra, OmegaConf, scikit-learn, GPyTorch, HPO-B, Dockstring,
PFNs4BO, and the BED baselines remain subject to their respective upstream
licenses. They are dependencies or separately obtained assets rather than code
relicensed by POLAR.
