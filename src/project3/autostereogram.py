import cv2
import numpy as np
import random

# Returns (depth_map, height, width)
def load_depth_map(path):
    d_map = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (d_map, d_map.shape[0], d_map.shape[1])

def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return (b, g, r)

def make_texture(shape=(32,32)):
    """ Create a texture of random colors.
        There's probably a better way to init a 2d array of random tuples...
    """
    texture = np.zeros((shape[0],shape[1], 3), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            color = random_color()
            texture[i][j] = (color[0], color[1], color[2])
    return texture

def make_autostereogram(depth_map, texture, scale):
    """
    Makes an autostereogram given a depthmap and a texture.
    Colors pixels with a corresponding pixel in the texture, with a displacement
    related to the depth found in the depth map
    """
    random.seed(None)
    height, width = depth_map.shape[0], depth_map.shape[1]
    pixel_count = 0
    colored = np.zeros((height, width), dtype=bool)
    sgram = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            color = texture[i%texture.shape[0]][j%texture.shape[1]]
            if j < width:
                sgram[i][j] = (color[0], color[1], color[2])
                colored[i][j] = True
                z = depth_map[i][j]
                d = (z + 150) * scale
                j = j + d
                pixel_count += 1
            if j >= width:
                j = 0
                while j < width and colored[i][j] == True:
                    j += 1  # Get first position j for row i that is uncolored

    t_asgram = make_transparent_autostereogram(depth_map, texture, int(scale/2))
    return sgram

def make_transparent_autostereogram(depth_map, texture, scale):
    height, width = depth_map.shape[0], depth_map.shape[1]
    pixel_count = 0
    colored = np.zeros((height, width), dtype=bool)
    sgram = np.zeros((height, width, 4), np.uint8)      # 4th channel is the alpha Channel
    for i in range(height):
        for j in range(width):
            color = texture[i%texture.shape[0]][j%texture.shape[1]]
            if j < width:
                sgram[i][j] = (color[0], color[1], color[2], 1)
                colored[i][j] = True
                z = depth_map[i][j]
                d = (z + 100) * scale                   # Use a smaller scale than used in non-transparent
                j = j + d
            if j >= width:
                j = 0
                while j < width and colored[i][j] == True:
                    j += 1  # Get first position j for row i that is uncolored

    # Make it transparent by going through the sgram and setting every 5th pixel to visible
    # And all others to transparent
    for i in range(height):
        for j in range(width):
            if pixel_count % 150 == 0:
                sgram[i][j] = (sgram[i][j][0], sgram[i][j][1], sgram[i][j][2], 1)
            else:
                sgram[i][j] = (222, 222, 222, 0.0)

    show_image(sgram, "transparent")
    return sgram




def show_image(image, description):
    cv2.imshow(description, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========== MAIN ==========
dm_name = 'shark-dm'
dm_file_type = '.jpg'
dm, h, w = load_depth_map(dm_name + dm_file_type)       # Load the depth map
show_image(dm, 'depth map')
texture = make_texture(shape=(int(h/10), int(w/10)))    # Make a random color texture for reference in autostereogram creation
show_image(texture, "texture")

sgram = make_autostereogram(dm, texture, 1)                # Make the autostereogram
show_image(sgram, 'sgram')
cv2.imwrite(dm_name + '-sgram.jpg', sgram)
cv2.imwrite('texture.jpg', texture)
