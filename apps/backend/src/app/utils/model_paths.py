from pathlib import Path

util_dir = Path(__file__).parent
app_dir = util_dir.parent
models_dir = app_dir / "models"

pose_model = {
  "lite_model" : models_dir / "pose_landmarker_lite.task",
  "full_model" : models_dir / "pose_landmarker_full.task",
  "heavy_model" : models_dir / "pose_landmarker_heavy.task"
}