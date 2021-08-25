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

def show_trainig_results(loss_meter, score_meter):
    _, axes = plt.subplots(1, 2, figsize=(15,5))

    axes[0].set_title("Loss")
    axes[0].plot(loss_meter.history)
    axes[0].plot(loss_meter.moving_average(0.9))

    axes[1].set_title("Scores")
    axes[1].plot(score_meter.history)
    axes[1].plot(score_meter.moving_average(0.9))

    # plt.title('Train metrics')
    plt.plot()

def show_loss(train_info):
    _, axes = plt.subplots(1, 2, figsize=(15,5))

    valid_iters = train_info['valid_iters']

    axes[0].set_title("Loss")
    axes[0].plot(valid_iters, train_info['train_loss_history'], '-o')
    axes[0].plot(valid_iters, train_info['valid_loss_history'], '-o')
    # axes[0].set_xticks(valid_iters)
    axes[0].legend(['train', 'val'], loc='upper right')
    
    axes[1].set_title("Scores")
    axes[1].plot(valid_iters, train_info['train_score_history'], '-o')
    axes[1].plot(valid_iters, train_info['valid_score_history'], '-o')
    axes[1].legend(['train', 'val'], loc='lower right')

    plt.plot()