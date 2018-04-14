from scipy.misc import toimage
import matplotlib.pyplot as plt

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb(data, i):
    img = toimage(data[0][i])
    bbox = data[1][i][0:4]    
    plt.imshow(img)
    plt.gca().add_patch(create_rect(bbox))