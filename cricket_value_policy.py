# -*- coding: utf-8 -*-
import numpy as np

p_out_min = [0.01, 0.02, 0.03, 0.1, 0.3]
p_out_max = [0.1, 0.2, 0.3, 0.5, 0.7]
p_run_max = 0.9
p_run_min = 0.3


runs = [1, 2, 3, 4, 6]


def get_probs(w, a):
    p_w = p_out_max[a] + (p_out_min[a] - p_out_max[a]) * ((w - 1) / 9)
    p_r = p_run_min + (p_run_max - p_run_min) * ((w - 1) / 9)
    return p_w, p_r


def Calculate_Value(V_in):
    Q_out = np.zeros((max_wickets, 5), dtype=float)
    V_out = np.zeros(np.size(V_in), dtype=float)
    shot = np.zeros(np.size(V_in), dtype=int)
    for i in range(1, np.size(V_in)):
        for a in range(np.size(runs)):
            [p_w, p_r] = get_probs(i, a)
            Q_out[i][a] = (1 - p_w) * (
                p_r * (runs[a] + V_in[i]) + (1 - p_r) * (0 + V_in[i])
            ) + p_w * V_in[i - 1]
        V_out[i] = np.max(Q_out[i, :])
        shot[i] = runs[np.argmax(Q_out[i, :])]
    return shot, V_out, Q_out


max_balls = 300
max_wickets = 11
V = np.zeros((max_balls, max_wickets), dtype=float)
Q = np.zeros((max_balls, max_wickets, 5), dtype=float)
PI = np.zeros((max_balls, max_wickets), dtype=int)
PI[max_balls - 1, :], V[max_balls - 1, :], Q[max_balls - 1, :] = Calculate_Value(
    np.zeros((max_wickets), dtype=float)
)
for i in range(max_balls - 2, -1, -1):
    PI[i, :], V[i, :], Q[i, :, :] = Calculate_Value(V[i + 1, :])

import csv
import numpy as np
from google.colab import drive

drive.mount("/content/drive")
np.savetxt("/content/drive/My Drive/test.csv", PI, delimiter=",")

print("Balls Left", "\t", end="")
for j in range(max_wickets):
    print(f"W={j}", "\t", end="")
print("\n")

for i in range(max_balls):
    print(300 - i, "\t", end="")
    for j in range(max_wickets):
        print(f"{V[i][j]:2.2f}", "\t", end="")
    print("\n")

for j in range(max_wickets):
    print(f"W={j}", "\t", end="")
print("\n")

for i in range(max_balls):
    for j in range(max_wickets):
        print(f"{V[i][j]:2.2f}", "\t", end="")
    print("\n")

for i in range(max_balls):
    for j in range(max_wickets):
        print(PI[i][j], "\t", end="")
    print("\n")
