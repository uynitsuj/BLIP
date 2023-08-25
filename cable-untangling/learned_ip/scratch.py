import numpy as np

#Compare crossing confidences in loose and tight knot

#loose knot
loose_crossings = np.load("loose_knot1_data/crossings.npy", allow_pickle=True)

#tight knot
tight_crossings = np.load("tight_knot1_data/crossings.npy", allow_pickle=True)

print("loose_crossings" , loose_crossings)
print("tight_crossings" , tight_crossings)