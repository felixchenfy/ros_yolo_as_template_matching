int32 N                 # Number of detected objects.
string label            # Nx1. Label of each object.
float32[] confidence    # Nx1. Confidence of each object. Range=[0, 1].
float32[] bbox          # Nx4. (x0, y0, x1, y1). Range=[0, 1].