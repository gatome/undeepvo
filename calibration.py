fx = 348.522264
fy = 348.449870
cx = 344.483596
cy = 188.808062
baseline = 0.118718331  # from T_cn_cnm1 (right relative to left)

P0 = [fx, 0, cx, 0,
      0, fy, cy, 0,
      0,  0,  1, 0]

P1 = [fx, 0, cx, -fx * baseline,
      0, fy, cy, 0,
      0,  0,  1, 0]

with open("rosario_dataset/sequences/01/calib.txt", "w") as f:
    f.write("P0: " + " ".join(f"{v:.9e}" for v in P0) + "\n")
    f.write("P1: " + " ".join(f"{v:.9e}" for v in P1) + "\n")

print("✅ calib.txt created for sequence 01")
