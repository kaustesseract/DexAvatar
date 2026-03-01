import numpy as np


BOF_body = np.array([
    [-120.0, -130.0,  -80.0, # left shoulder
     -120.0,   0.0,  -80.0, # right shoulder
     -120.0, -160.0, -140.0, # left elbow
     -120.0,    0.0,  -140.0, # right elbow
     -120.0,  -50.0,  -90.0, # left wrist
     -120.0,  -50.0,  -90.0, # right wrist
     ],
     [90.0,    0.0,   80.0, # left shoulder
      90.0, 130.0,   80.0, # right shoulder
      90.0,    0.0,   140.0, # left elbow
      90.0,  160.0,    140.0, # right elbow
      90.0,   50.0,   90.0, # left wrist
      90.0,   50.0,   90.0]]) / 180 * np.pi


