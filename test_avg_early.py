from utils import LevenshteinDistancePlusAvgFramesEarly

a = [1, 2, 3, 4, 5]
b = [2, 3, 4, 5]
c = [1, 1, 2, 2, 3, 4, 5]

a_end_frames = [100, 200, 300, 400, 500]
b_detect_frames = [150, 290, 450, 500]
c_detect_frames = [10, 60, 180, 190, 260, 340, 480]

print(f'{LevenshteinDistancePlusAvgFramesEarly(a, b, a_end_frames, b_detect_frames)} == (1, 2.5)')
print(f'{LevenshteinDistancePlusAvgFramesEarly(a, c, a_end_frames, c_detect_frames)} == (2, 46)')

a = [23, 56, 52, 17, 39, 15, 30, 81, 26, 0]
b = [23, 57, 52, 77, 39,     30,     26]
a_end_frames =    [197, 267, 329, 436, 587, 646, 709, 1018, 1253, 1359]
b_detect_frames = [199, 278, 333, 520, 590,      719,       1257]

print(f'{LevenshteinDistancePlusAvgFramesEarly(a, b, a_end_frames, b_detect_frames)} == (1, 2.5)')