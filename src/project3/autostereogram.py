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
        print(" row %d " % i)
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

    return sgram

def put_overlay(image, overlay, alpha):
    image = cv2.addWeighted(image, alpha, overlay, .75, -20)
    return image


# ========== IO ===================
def show_image(image, description):
    cv2.imshow(description, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========== ========== ========= ========== ========== MAIN ========== ========== ========== ========== ==========
dm_name = 'shark-dm'
dm_file_type = '.jpg'

dm, h, w = load_depth_map(dm_name + dm_file_type)       # Load the depth map
show_image(dm, 'depth map')
texture = make_texture(shape=(int(h/10), int(w/10)))    # Make a random color texture for reference in autostereogram creation
show_image(texture, "texture")

print( "MAKING AUTOSTEREOGRAM" )
sgram = make_autostereogram(dm, texture, 2)                # Make the autostereogram
show_image(sgram, 'sgram')


print( "MAKING OVERLAY" )
overlay = make_autostereogram(dm, texture, .1)             # Make another, but with a smaller shift and transparent
sgram_with_overlay = put_overlay(sgram, overlay, 0.4)      # and stack em and show em
show_image(sgram, 'sgram with overlay')

cv2.imwrite(dm_name + '-sgram.jpg', sgram)                 # Save em
cv2.imwrite('texture.jpg', texture)
cv2.imwrite(dm_name + '-sgram-with-overlay.jpg', sgram_with_overlay)
