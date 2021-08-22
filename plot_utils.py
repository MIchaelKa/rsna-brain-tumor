import matplotlib.pyplot as plt

def show_image_3d(image_3d):
    image_3d = image_3d[0]

    size = 2
    num_images = image_3d.shape[0]
    ncol = 10
    nrow = int(num_images / ncol) + 1
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * size, nrow * size))
    axes = axes.flatten()
    
    for i in range(num_images):
        ax = axes[i]
        image = image_3d[i]
        
        ax.imshow(image, cmap='bone')
        ax.axis('off')
    
    print(f'shape: {image_3d.shape}')
    plt.show()