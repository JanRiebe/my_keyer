import numpy as np
import cv2


# picture variables
canvas = (1080, 1920)
fg = None
bg = None


# manipulation variables
a_gain = 1
a_lift = 0
a_blur = 0
fg_pos = [0, 0]
bg_pos = [0, 0]


# setting up the viewer window
cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)

def update_viewer(alpha):
    """ draws the original image, the matte and the comp into a side by side viewer
    """

    # creating matte image to be displayed
    matte = np.ones([fg.shape[0], fg.shape[1], 3], 'float')
    matte[:, :, 0] = alpha
    matte[:, :, 1] = alpha
    matte[:, :, 2] = alpha

    #TODO rework to allow negative positions
    # cutting foreground to view frustum
    fg_canvas = np.zeros([canvas[0], canvas[1], 3], 'float')
    height_cut = min(fg_pos[0] + fg.shape[0], canvas[0])
    width_cut = min(fg_pos[1] + fg.shape[1], canvas[1])
    fg_cut = fg[0: canvas[0]-fg_pos[0], 0: canvas[1]-fg_pos[1]]
    fg_canvas[fg_pos[0]:height_cut, fg_pos[1]:width_cut] = fg_cut

    # cutting matte to view frustum
    matte_canvas = np.zeros([canvas[0], canvas[1], 3], 'float')
    matte_cut = matte[0: canvas[0]-fg_pos[0], 0: canvas[1]-fg_pos[1]]
    matte_canvas[fg_pos[0]:height_cut, fg_pos[1]:width_cut] = matte_cut

    # cutting background to view frustum
    bg_canvas = np.zeros([canvas[0], canvas[1], 3], 'float')
    height_cut = min(bg_pos[0] + bg.shape[0], canvas[0])
    width_cut = min(bg_pos[1] + bg.shape[1], canvas[1])
    bg_cut = bg[0: canvas[0]-bg_pos[0], 0: canvas[1]-bg_pos[1]]
    bg_canvas[bg_pos[0]:height_cut, bg_pos[1]:width_cut] = bg_cut

    # calculating comp image to be displayed
    comp = fg_canvas*matte_canvas + bg_canvas*(1-matte_canvas)

    # show viewer
    viewer = np.concatenate((fg_canvas, matte_canvas, comp), axis=1)
    cv2.imshow("viewer", viewer)

    return comp, matte_canvas


def create_matte():
    """ creates a matte from the foreground image
    :return: a matrix representing the alpha channel with values between 0.0 and 1.0
    """

    # Matte generation:
    # separating channels
    b = fg[:, :, 0]
    g = fg[:, :, 1]
    r = fg[:, :, 2]

    # calculating matte
    a = 1 - ((g - b) + (g - r))
    a = a * a_gain
    a = a + a_lift

    # blurring matte
    if a_blur > 0:
        a = cv2.blur(a, (a_blur, a_blur))

    return np.clip(a, 0.0, 1.0)





# reading in foreground
fg_path = input("Please enter the path of the foreground image: ")
fg = cv2.imread(fg_path).astype(np.float32)/255.0
while fg is None:
    print("Could not load image from this path.")
    fg_path = input("Please enter a valid path of the image: ")
    fg = cv2.imread(fg_path)

# reading in background
bg_path = input("Please enter the path of the background image: ")
bg = cv2.imread(bg_path).astype(np.float32)/255.0
while bg is None:
    print("Could not load image from this path.")
    bg_path = input("Please enter a valid path of the image: ")
    bg = cv2.imread(bg_path)


# input loop
user_input = ""
while user_input != "q":

    # change computation parameters depending on user input
    if user_input == "h":
        print("type 'gain' to change alpha gain")
        print("type 'lift' to change alpha lift")
        print("type 'tl fg' to translate the foreground image")
        print("type 'tl bg' to translate the background image")
    elif user_input == "gain":
        a_gain = float(input("alpha gain: "))
    elif user_input == "lift":
        a_lift = float(input("alpha lift: "))
    elif user_input == "blur":
        a_blur = int(input("alpha blur: "))
    elif user_input == "tl fg":
        fg_pos[1] = max(0, int(input("translate foreground x: ")))
        fg_pos[0] = max(0, int(input("translate foreground y: ")))
    elif user_input == "tl bg":
        bg_pos[1] = max(0, int(input("translate background x: ")))
        bg_pos[0] = max(0, int(input("translate background y: ")))

    alpha = create_matte()

    update_viewer(alpha)

    cv2.waitKey(1)
    user_input = input("Chance values (h: help, q: quit): ")


# cleanup
cv2.destroyAllWindows()
