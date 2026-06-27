# SOMA-BVH zu SMPL-X: funktionierender Ablauf

Diese Notiz beschreibt den validierten Ablauf für BONES-SEED/SOMA-Uniform-BVH-Dateien. Sie hält insbesondere die Fehler fest, die zu seitlichen Körpern, verdrehten Posen sowie deformierten Köpfen und Füßen geführt haben.

## Grundregeln

- Quelldateien niemals löschen oder überschreiben.
- Für korrigierte Ergebnisse immer neue, eindeutig benannte Dateien erzeugen.
- Für nachgelagerte Mesh-Ausgaben `reconstructed_vertices` aus dem Ergebnis des Pose-Converters verwenden. Die SMPL-X-Vertices nicht nochmals unabhängig aus exportierten Poseparametern erzeugen, weil dadurch Abweichungen entstehen können.
- Vor einer vollständigen Konvertierung immer zuerst einen einzelnen Frame prüfen.

## Eingabedaten dieses Beispiels

BVH:

```text
<BONES_SEED_ROOT>\soma_uniform\soma_uniform\bvh\230317\nailing_wall_R_003__A282.bvh
```

Die Datei ist bytegleich mit der BONES-SEED-SOMA-Uniform-Version. Deshalb wird die gemeinsame Uniform-Form verwendet:

```text
<BONES_SEED_ROOT>\soma_shapes\soma_base_fit_mhr_params.npz
```

SMPL-X-Modell:

```text
assets\SMPLX\SMPLX_NEUTRAL.npz
```

## 1. BVH zu SOMA-NPZ

Verwendetes Werkzeug:

```text
tools/convert_bones_seed_bvh_to_soma.py
```

Beispiel:

```powershell
python `
  tools\convert_bones_seed_bvh_to_soma.py `
  "<BONES_SEED_ROOT>\soma_uniform\soma_uniform\bvh\230317\nailing_wall_R_003__A282.bvh" `
  outputs\nailing_wall\nailing_wall_R_003__A282_uniform_absolute_soma.npz `
  --shape "<BONES_SEED_ROOT>\soma_shapes\soma_base_fit_mhr_params.npz"
```

Erwartetes Ergebnis:

- 1340 Frames
- ungefähr 120.005 fps
- `poses`: `(1340, 77, 3, 3)`
- `transl`: `(1340, 3)` in Metern
- kein virtueller `Root` in `poses`; die Rotationen beginnen bei `Hips`

### Wichtig: Rotationen sind absolut

Die BVH-Kanalrotationen enthalten die SOMA-Gelenkorientierung bereits. Sie müssen daher als **absolute lokale Rotationen** gespeichert werden:

```python
absolute_pose = True
```

Wenn sie fälschlich als relativ gespeichert werden, wendet `SOMALayer` die Gelenkorientierung ein zweites Mal an. Das erzeugt einen seitlichen Körper und stark verdrehte Posen.

Die Euler-Kanäle werden in der Reihenfolge verarbeitet, in der sie im BVH stehen. Bei dieser Datei ist das normalerweise:

```text
Zrotation Yrotation Xrotation
```

Die Matrizen werden entsprechend in Kanalreihenfolge multipliziert. Die BVH-Einheit Zentimeter wird für `transl` mit `0.01` in Meter umgerechnet.

Der virtuelle `Root` ist in dieser Aufnahme neutral. Allgemein wird sein Transform trotzdem korrekt in `Hips` gefaltet:

```python
hips_rotation = root_rotation @ hips_rotation
hips_translation = root_translation + root_rotation @ hips_local_translation
```

## 2. SOMA zu SMPL-X

Verwendetes Werkzeug:

```text
tools/pose_converter.py
```

Vor der vollständigen Bewegung zuerst einen Frame konvertieren:

```text
--max-frames 1
```

Danach kann die vollständige Bewegung konvertiert werden. Ohne CUDA-Gerät wird CPU verwendet. Falls Warps Standardcache nicht beschreibbar ist, kann ein Cache im Workspace gesetzt werden:

```powershell
python -c "import os,sys,runpy; os.environ['TEMP']=r'.tmp'; os.environ['TMP']=r'.tmp'; import warp as wp; wp.config.kernel_cache_dir=r'.warp-cache'; sys.argv=['tools/pose_converter.py','--source','soma','--target','smplx','--input',r'outputs\nailing_wall\nailing_wall_R_003__A282_uniform_absolute_soma.npz','--output',r'outputs\nailing_wall\nailing_wall_R_003__A282_uniform_absolute_neutral_target_smplx.npz','--data-root',r'assets','--device','cpu']; runpy.run_path('tools/pose_converter.py', run_name='__main__')"
```

### Wichtig: SOMA-Identität ist nicht SMPL-X-Beta

SOMA/MHR-Identitätskoeffizienten und SMPL-X-Betas liegen in verschiedenen Parameterbereichen. Die ersten zehn SOMA/MHR-Werte dürfen **nicht** abgeschnitten und als SMPL-X-Betas verwendet werden.

Das frühere Verhalten erzeugte unter anderem:

- einen unnatürlich großen oder deformierten Kopf,
- deformierte Füße,
- unpassende Körperproportionen,
- einen deutlich höheren Vertexfehler.

Für SOMA/MHR zu SMPL-X wird deshalb standardmäßig die neutrale SMPL-X-Identität verwendet, sofern keine explizit passend geschätzten SMPL-X-Betas vorliegen:

```python
target_identity = zeros
```

Die Korrektur befindet sich in:

```text
soma/smpl/transfer.py
```

Identitätskoeffizienten werden nur wiederverwendet, wenn Quell- und Zielmodell tatsächlich denselben kompatiblen Identitätsraum besitzen.

## 3. Ergebnis prüfen

Vollständiges korrigiertes Ergebnis:

```text
outputs\nailing_wall\nailing_wall_R_003__A282_uniform_absolute_neutral_target_smplx.npz
```

Wichtige Arrays:

- `target_rotations`: SMPL-X-Gelenkrotationen
- `target_root_translation`: globale SMPL-X-Wurzeltranslation
- `source_vertices`: SOMA-Quellmesh
- `fit_vertices`: auf SMPL-X-Topologie übertragene Zielpunkte
- `reconstructed_vertices`: tatsächlich aus den ermittelten SMPL-X-Parametern rekonstruierte Vertices
- `per_vertex_error`: Abstand zwischen Fit-Ziel und Rekonstruktion

Validierte Größe:

```text
reconstructed_vertices: (1340, 10475, 3)
```

Nach der Identitätskorrektur sank der mittlere Fehler ungefähr von `0.0388 m` auf `0.0140 m`. Der maximale Fehler der vollständigen Bewegung lag bei ungefähr `0.0921 m`.

### Geometrische Plausibilitätsprüfung

Die aus dem korrigierten SOMA-NPZ berechneten Gelenkpositionen wurden mit direkter BVH-Forward-Kinematics verglichen:

- RMS-Abweichung über die Gelenke: ungefähr 8 mm
- Füße und Zehen: ungefähr 4 bis 5 mm Abweichung
- Kopf: ungefähr 1 cm Abweichung

Damit wurde bestätigt, dass BVH-Parsing, Achsen, Translation und Gelenkhierarchie korrekt interpretiert werden. Die auffällige Form von Kopf und Füßen entstand anschließend durch die falsche Übernahme der Identitätskoeffizienten.

## 4. Häufige Fehler

1. `absolute_pose=False` für BONES-SEED-BVH verwenden: Gelenkorientierung wird doppelt angewendet.
2. SOMA/MHR-Identitätswerte als SMPL-X-Betas verwenden: Kopf, Füße und Körperform werden deformiert.
3. Nur die Skelettpositionen prüfen: falsche Gelenktwists können bei ähnlichen Gelenkpositionen trotzdem das Mesh verdrehen.
4. Einen vollständigen Lauf starten, bevor Frame 0 geprüft wurde: kostet unnötig Zeit und erzeugt sehr große fehlerhafte Dateien.
5. `fit_vertices` als endgültige Bewegung behandeln: für nachgelagerte Verarbeitung immer `reconstructed_vertices` verwenden.
6. Alte Ergebnisse überschreiben: korrigierte Varianten mit eindeutigen Namen speichern.
