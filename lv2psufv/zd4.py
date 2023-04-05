import numpy as np
import matplotlib.pyplot as plt

def chessboard(square_size, num_rows, num_cols):
    black_square = np.zeros((square_size, square_size))
    white_square = np.ones((square_size, square_size)) * 255

    rows = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            if (i+j) % 2 == 0:
                row.append(black_square)
            else:
                row.append(white_square)
        rows.append(np.hstack(row))

    img = np.vstack(rows).astype(np.uint8)
    return img

# Primjer poziva funkcije
img = chessboard(20, 4, 5)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()