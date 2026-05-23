# OpenSim Markers To SOMA Skeleton Alignment

This note documents the marker/skeleton mesh pipeline used in this workspace.
It creates readable, named GLB marker meshes from an OpenSim `.osim` file, creates
a named SOMA stick skeleton from BVH, and applies the extra arm/hand marker
adjustments needed for the T-pose alignment.

Run commands from:

```powershell
cd C:\Users\stlec\Projects\SOMA-X
```

## 1. Create the SOMA skeleton stick mesh

This reads the BVH hierarchy and uses frame `0` forward kinematics, so the
skeleton is not flat.

```powershell
.\.venv\Scripts\python.exe tools\create_bvh_stick_mesh.py `
  C:\Users\stlec\Projects\bones-seed\soma_shapes\soma_base_rig\soma_base_skel_minimal.bvh `
  --out-dir out `
  --name soma_base_skel_minimal
```

Important output:

```text
out\soma_base_skel_minimal_stick_named.glb
out\soma_base_skel_minimal_stick_manifest.csv
out\soma_base_skel_minimal_stick_manifest.json
```

The GLB keeps readable names like `joint_001_Hips` and
`bone_001_002_Hips__to__Spine1`.

## 2. Create marker-only mesh from the OpenSim model

This extracts only the OpenSim `MarkerSet` markers from the `.osim` file. It
exports one named sphere per marker and writes a CSV/JSON manifest with the
original body-local marker position and computed default-pose position.

```powershell
.\.venv\Scripts\python.exe tools\create_osim_marker_mesh.py `
  "C:\Users\stlec\Downloads\Arm swing simulation data\arm_swing\Adjusted_ULBmodel.osim" `
  --out-dir out `
  --name Adjusted_ULBmodel
```

Important output:

```text
out\Adjusted_ULBmodel_markers_named.glb
out\Adjusted_ULBmodel_markers_manifest.csv
out\Adjusted_ULBmodel_markers_manifest.json
```

Marker GLB names look like:

```text
marker_000_STRN__body_thorax
marker_001_RSHO__body_thorax
marker_006_RUPA__body_humerus
```

## 3. Transform all OpenSim markers into the SOMA alignment space

This applies the full-marker transform we settled on:

- uniform scale: `1.0629`
- right-handed Y rotation: `-90` degrees
- translation: `+0.12` along Z

```powershell
.\.venv\Scripts\python.exe tools\transform_marker_mesh.py `
  out\Adjusted_ULBmodel_markers_named.glb `
  out\Adjusted_ULBmodel_markers_manifest.json `
  --out-glb out\Adjusted_ULBmodel_markers_named_scale1p0629_yneg90_zplus0p12.glb `
  --out-json out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12.json `
  --out-csv out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12.csv `
  --scale 1.0629 `
  --y-degrees -90 `
  --translate 0 0 0.12
```

Important output:

```text
out\Adjusted_ULBmodel_markers_named_scale1p0629_yneg90_zplus0p12.glb
out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12.csv
out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12.json
```

## 4. Rotate and shift arm/hand markers into the T-pose

This rotates only the arm/hand markers in the XY plane around global Z, using
the first sign convention that looked correct:

- right arm/hand: `-90` degrees around `RSHO`
- left arm/hand: `+90` degrees around `LSHO`

Then it applies the final arm-only translation:

- right arm/hand: `+0.05 X`, `-0.14 Y`, `-0.055 Z`
- left arm/hand: `-0.05 X`, `-0.14 Y`, `-0.055 Z`

```powershell
.\.venv\Scripts\python.exe tools\pose_arm_markers_tpose.py `
  out\Adjusted_ULBmodel_markers_named_scale1p0629_yneg90_zplus0p12.glb `
  out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12.json `
  --out-glb out\Adjusted_ULBmodel_markers_named_scale1p0629_yneg90_zplus0p12_arm_tpose_xy_firstsign_arms_down0p14_zneg0p055_xside0p05.glb `
  --out-json out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12_arm_tpose_xy_firstsign_arms_down0p14_zneg0p055_xside0p05.json `
  --out-csv out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12_arm_tpose_xy_firstsign_arms_down0p14_zneg0p055_xside0p05.csv `
  --right-degrees -90 `
  --left-degrees 90 `
  --right-arm-translate 0.05 -0.14 -0.055 `
  --left-arm-translate -0.05 -0.14 -0.055
```

Final marker output to use:

```text
out\Adjusted_ULBmodel_markers_named_scale1p0629_yneg90_zplus0p12_arm_tpose_xy_firstsign_arms_down0p14_zneg0p055_xside0p05.glb
out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12_arm_tpose_xy_firstsign_arms_down0p14_zneg0p055_xside0p05.csv
out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12_arm_tpose_xy_firstsign_arms_down0p14_zneg0p055_xside0p05.json
```

## Final files for alignment

Use these two GLBs together in your viewer/alignment tool:

```text
out\soma_base_skel_minimal_stick_named.glb
out\Adjusted_ULBmodel_markers_named_scale1p0629_yneg90_zplus0p12_arm_tpose_xy_firstsign_arms_down0p14_zneg0p055_xside0p05.glb
```

Use these manifests if you need names and coordinates in code:

```text
out\soma_base_skel_minimal_stick_manifest.csv
out\Adjusted_ULBmodel_markers_manifest_scale1p0629_yneg90_zplus0p12_arm_tpose_xy_firstsign_arms_down0p14_zneg0p055_xside0p05.csv
```

## Scripts

```text
tools\create_bvh_stick_mesh.py       # BVH skeleton -> named stick GLB
tools\create_osim_marker_mesh.py     # OpenSim .osim MarkerSet -> named marker GLB
tools\transform_marker_mesh.py       # whole-marker scale/rotate/translate
tools\pose_arm_markers_tpose.py      # arm/hand marker rotation and side-specific shifts
```
