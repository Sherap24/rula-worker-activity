# Data Acquisition Guide

This project stores **large datasets outside the OneDrive repository**. The code repository lives at:

`C:\Users\asahi\OneDrive\Desktop\RULA\Project`

Raw archives and extracted datasets belong under a separate data root controlled by the `RULA_DATA_ROOT` environment variable.

---

## 1. Why datasets live outside OneDrive

- CWPV is approximately **11.6 GB**; CML is approximately **1.2 GB**.
- OneDrive sync conflicts with large binary files, partial uploads, and git operations.
- Research data should remain local, gitignored, and reproducibly referenced via manifests — not committed.

---

## 2. Create external directories

Create this layout (adjust the drive letter if needed):

```
C:\Users\asahi\Datasets\RULA\
    cml\
        archives\
        extracted\
    cwpv\
        archives\
        extracted\
    cma\
    kaggle_construction\
    local_smartphone\
```

PowerShell:

```powershell
$root = "C:\Users\asahi\Datasets\RULA"
@(
    "cml\archives", "cml\extracted",
    "cwpv\archives", "cwpv\extracted",
    "cma", "kaggle_construction", "local_smartphone"
) | ForEach-Object { New-Item -ItemType Directory -Path (Join-Path $root $_) -Force }
```

---

## 3. Set `RULA_DATA_ROOT` temporarily (PowerShell)

```powershell
$env:RULA_DATA_ROOT = "C:\Users\asahi\Datasets\RULA"
```

Verify:

```powershell
python -m worker_activity audit-data
```

---

## 4. Set `RULA_DATA_ROOT` persistently (Windows)

**User environment variable (recommended):**

1. Open *Settings → System → About → Advanced system settings → Environment Variables*.
2. Under *User variables*, click *New*.
3. Name: `RULA_DATA_ROOT`
4. Value: `C:\Users\asahi\Datasets\RULA`
5. Restart the terminal (or Cursor) after saving.

Alternatively, copy `.env.example` to `.env` and load it in your shell session. **Do not commit `.env`.**

---

## 5. CML — download and placement

| Item | Value |
|------|-------|
| Official Figshare | https://figshare.com/articles/dataset/Construction_Motion_Data_Library_An_Integrated_Motion_Dataset_for_On_Site_Activity_Recognition/20480787 |
| DOI | https://doi.org/10.6084/m9.figshare.20480787 |
| Supporting repo | https://github.com/YUANYUAN2222/Integrated-public-3D-skeleton-form-CML-library |
| License | CC BY 4.0 |
| Approx. size | 1.18 GB |

**Download first.** CML is the primary skeleton and activity-label reference.

1. Download the Figshare archive manually.
2. Save the archive to:

   `C:\Users\asahi\Datasets\RULA\cml\archives\`

3. Extract contents to:

   `C:\Users\asahi\Datasets\RULA\cml\extracted\`

4. **Do not run 2D video pose estimation on CML** — it provides pre-extracted 3D skeleton data.

---

## 6. CWPV — download and placement

| Item | Value |
|------|-------|
| Official Figshare | https://figshare.com/articles/dataset/CWPV_A_Working_Postures_of_the_Construction_Working_Postures_Videos_dataset/27907818 |
| DOI | https://doi.org/10.6084/m9.figshare.27907818 |
| License | CC BY 4.0 (Version 3, 2026-02-18) |
| Approx. size | 11.61 GB |

**Primary Week 3 video source.**

1. Ensure **≥ 15 GB free disk space** before downloading.
2. Download the Figshare archive manually.
3. Save the archive to:

   `C:\Users\asahi\Datasets\RULA\cwpv\archives\`

4. Extract contents to:

   `C:\Users\asahi\Datasets\RULA\cwpv\extracted\`

5. **Retain the CWPV README** included with the dataset for subject, task, and camera-layout documentation.
6. IMU files are available but **outside the initial vision baseline**.

---

## 7. Deferred sources

| Source | Reason deferred |
|--------|-----------------|
| **CMA** | Baidu Netdisk access, Python 3.6 legacy repo, non-commercial restriction — optional Week 7 RGB comparison |
| **ICON-Pose** | Egocentric; no verified bulk download in registry — future extension |
| **Kaggle construction** | CC0 on Kaggle page, but labels/subjects/splits not locally verified — optional after CML + CWPV inspection |
| **Local smartphone** | See §7a (Week 5 domain transfer) |

### 7a. Local smartphone capture (Week 5)

Place self-filmed clips under:

```text
%RULA_DATA_ROOT%\local_smartphone\
```

**Filename convention** (label is parsed from the stem):

```text
{canonical_activity}_{nn}.mp4
```

Examples: `squatting_01.mp4`, `carrying_02.mp4`.

Allowed `canonical_activity` values match the CWPV baseline classes:

- `carrying`
- `kneeling`
- `lifting_lowering`
- `overhead_work_reaching`
- `squatting`

**Consent:** Prefer self-capture for research. If filming others, obtain informed consent before recording. Do not commit raw phone videos to git; keep them only under `RULA_DATA_ROOT`.

**Evaluation only:** Phone clips measure domain transfer of CWPV-trained models. Do not use them to train or tune against the held-out CWPV test set.

---

## 8. Licensing and citation obligations

- **CML** and **CWPV**: CC BY 4.0 — attribution required in publications and derived works.
- **CMA**: research-only / non-commercial — verify terms before use.
- **Kaggle**: page shows CC0 — still verify content after local inspection.
- Cite official DOIs and dataset papers in the research manuscript.
- **Never commit** raw archives, extracted videos, or skeleton files to git.

---

## 9. After download — verify with CLI

```powershell
$env:RULA_DATA_ROOT = "C:\Users\asahi\Datasets\RULA"
python -m worker_activity audit-data
python -m worker_activity build-inventory
```

Outputs:

- `data/manifests/clip_inventory.csv`
- `data/manifests/clip_inventory.parquet`
- `reports/dataset_inventory.md`

---

## 10. Raw data policy

- Archives → `{RULA_DATA_ROOT}/{source}/archives/`
- Extracted → `{RULA_DATA_ROOT}/{source}/extracted/`
- Manifests and reports → inside the git repository under `data/manifests/` and `reports/`
- If `RULA_DATA_ROOT` is unset, CLI commands print warnings and produce empty inventories — **no stack traces**.
