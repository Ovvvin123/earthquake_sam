# Local artifacts

Large datasets and model weights are intentionally excluded from ordinary Git history. After cloning the repository, place the following files at these paths.

| Artifact | Local path | SHA-256 |
|---|---|---|
| SAM-2 small backbone | `SAM_code_extracted/SAM code/sam2_hiera_small.pt` | `95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3` |
| California v3 dataset | `SAM_code_extracted/SAM code/earth_dataset_v3/faults.h5` | `171b33339d43ae48f5ab166945dc7c06bff3ca9c6722a7ac2ea18f12c9ae1c38` |
| Historical v5 model | `SAM_code_extracted/SAM code/train_v5_local/model.pth` | `dbd2a0acfe9b729205196afa51cdfd06dd68ec2a799630c651318ed11efc5ab7` |

Verify a copied artifact on Ubuntu with:

```bash
sha256sum "path/to/file"
```
