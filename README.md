# Test yolov8 for gaze tracking

## Note

- May need to change the CUDA version in `environment.yml` depend on your computer. Or just use CPU pytorch.

- Use `conda env create -f environment.yml` to install and create env. THEN use `pip install ultralytics` (If this was put in the environment.yml file, it will freeze the install process)

- If install freeze,try to install it one by one

## Link

- <https://docs.ultralytics.com/quickstart/#install-ultralytics>

- <https://docs.ultralytics.com/modes/train/#train-settings>

- <https://github.com/ultralytics/ultralytics?tab=readme-ov-file>

- Youtube tutorial: <https://www.youtube.com/watch?v=LNwODJXcvt4>

- Dataset: <https://universe.roboflow.com/pupiup-rjvfv/cau001/dataset/1>

- Trained model (low quality, don't expect much): <https://drive.google.com/file/d/13mIAXKPk6kJ8HmryrdM3k6FbiSb51-yv/view?usp=sharing>

## FAQ

- Got RuntimeError `freeze_support()`?: <https://stackoverflow.com/questions/75111196/yolov8-runtimeerror-an-attempt-has-been-made-to-start-a-new-process-before-th>

- For list of supported CUDA version on your GPU, look for "GPUs supported" section in here: <https://en.wikipedia.org/wiki/CUDA>

