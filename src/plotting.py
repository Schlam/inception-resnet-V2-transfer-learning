import matplotlib.pyplot as plt

# Show a few example images

plt.figure(figsize=(8,5),dpi=100)

example1, example2 = images[-1], images[-2]

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.imshow(example1)
ax2.imshow(example2);



# Plot the model outputs if they aren't 1D tensors

if LITE:

    # Compare bird images to the sum of their pooling outputs
    BIRD = 12

    plt.figure(figsize=(8,5),dpi=100)

    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    ax1.imshow(images[BIRD])
    ax2.imshow(pooling_outputs[BIRD].sum(axis=-1))
