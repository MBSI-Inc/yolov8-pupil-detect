# Tutorial on training custom YOLO model

## Step 1: Get a dataset

- Go here <https://universe.roboflow.com/> and search for something suitable to use.
- Download as YOLOv8 format (you should get a zip file).

## Step 2: Training a new model

- Examine the `step2_training.py` file. The file also included comments explain the code. Run with `python step2_training.y` command.

## Step 3: Predict on an image

- Create a folder, add a 640x640 image on it (ideally with something relevant to your training data).
- Read and run the `step3_predict_img.py` file.

## Step 4: Predict on camera

- Make sure you have a camera connected and the weight file.
- If you didn't have the weight, read the README file.
- If you don't want to read the README file, here is the link for the model <https://drive.google.com/file/d/13mIAXKPk6kJ8HmryrdM3k6FbiSb51-yv/view?usp=sharing>
- Observe and execute the `step4_predict_camera.py` file.
