import numpy as np

# Step 1: Create mock MFCC data (10 time steps, 3 coefficients each)
np.random.seed(0)
utterance = np.round(np.random.rand(10, 3), 2)  # Shape: (10, 3)

print("Original MFCC features:")
print(utterance)

# Step 2: Define a function to perform the stacking with mirroring
def stack_features_with_mirroring(features, context=3):
    T, D = features.shape
    padded = np.concatenate([
        features[context:0:-1],  # Mirror the first `context` frames
        features,
        features[-2:-context-2:-1]  # Mirror the last `context` frames
    ], axis=0)

    # Now extract stacked windows
    stacked = []
    for t in range(T):
        window = padded[t : t + 2 * context + 1]  # 7 frames
        stacked.append(window.flatten())
    return np.stack(stacked)

# Step 3: Apply and print
stacked = stack_features_with_mirroring(utterance)
print("Original shape:", utterance.shape)
print("Stacked shape:", stacked.shape)
print("Example stacked frame at t=0:")
print(stacked[0].reshape(7, 3))  # Unflatten for visual clarity
print(stacked[-1].reshape(7, 3))  # Unflatten for visual clarity
