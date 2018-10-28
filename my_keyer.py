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
eb_size = 0         # width of the edge blur
lw_size = 0         # width of the light wrap outline
lw_intensity = 0
despill = True

# setting up the viewer window
cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)


def update_viewer(alpha):
    """ draws the original image, the matte and the comp into a side by side viewer
    """

    #spill reduction
    fg_despilled = fg.copy()
    if despill:
        # if green bigger than average red and blue
        # lower green to average
        # increase r and b by a total of the amount g was reduced relative to each other
        b = fg_despilled[:, :, 0]
        g = fg_despilled[:, :, 1]
        r = fg_despilled[:, :, 2]
        average = (b+r)/2
        g = np.maximum(g, average)
        g_diff = g - average
        #br_relation = b/r   #TODO not right, use this below
        b += g_diff/2
        r += g_diff/2
        fg_despilled[:, :, 0] = b
        fg_despilled[:, :, 1] = g
        fg_despilled[:, :, 2] = r
        print("spill reduction applied")
        print(despill)


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
    fg_cut = fg_despilled[0: canvas[0]-fg_pos[0], 0: canvas[1]-fg_pos[1]]
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

    # edge blur
    if eb_size > 0:
        res, matte_binary = cv2.threshold(matte_canvas[:, :, 0], 0.5, 1.0, cv2.THRESH_BINARY_INV)
        kernel = np.ones((eb_size, eb_size), 'float')
        edge = cv2.dilate(matte_binary, kernel) - matte_binary
        edge_img = np.zeros([canvas[0], canvas[1], 3], 'float')
        edge_img[:, :, 0] = edge
        edge_img[:, :, 1] = edge
        edge_img[:, :, 2] = edge
        edge_img = cv2.blur(edge_img, (eb_size, eb_size))

        comp_blurred = cv2.blur(comp, (eb_size, eb_size))
        comp = comp*(1-edge_img)+comp_blurred*edge_img

    # light wrap
    if lw_size > 0 and lw_intensity > 0:
        res, matte_binary = cv2.threshold(matte_canvas[:, :, 0], 0.5, 1.0, cv2.THRESH_BINARY_INV)
        kernel = np.ones((lw_size, lw_size), 'float')
        edge = cv2.dilate(matte_binary, kernel) - cv2.erode(matte_binary, kernel)
        edge = cv2.blur(edge, (lw_size, lw_size))
        edge *= (1-matte_binary)
        light_wrap = np.zeros([canvas[0], canvas[1], 3], 'float')
        light_wrap[:, :, 0] = edge
        light_wrap[:, :, 1] = edge
        light_wrap[:, :, 2] = edge
        bg_blurred = cv2.blur(bg_canvas, (lw_size, lw_size))
        light_wrap *= bg_blurred
        comp = np.maximum(comp, light_wrap*lw_intensity)

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

    #TODO try to use adaptive threshold and blur to even out the green channel before keying


    # calculating matte
    a = 1 - ((g - b) + (g - r))
    a = a * a_gain
    a = a + a_lift

    # blurring matte
    if a_blur > 0:
        a = cv2.blur(a, (a_blur, a_blur))

    return np.clip(a, 0.0, 1.0)





# reading in foreground
fg_path = "C:\\Users\\Jan\\Pictures\\Greenscreen_Sasha.jpg"  #input("Please enter the path of the image: ")
fg = cv2.imread(fg_path).astype(np.float32)/255.0
while fg is None:
    print("Could not load image from this path.")
    fg_path = input("Please enter a valid path of the image: ")
    fg = cv2.imread(fg_path)

# reading in background
bg_path = "C:\\Users\\Jan\\Pictures\\Bussard1495.jpg"  #input("Please enter the path of the image: ")
bg = cv2.imread(bg_path).astype(np.float32)/255.0

#while bg is None:
#    print("Could not load image from this path.")
#    fg_path = input("Please enter a valid path of the image: ")
#    fg = cv2.imread(bg_path)






# input loop
user_input = ""
while user_input != "q":

    # change computation parameters depending on user input
    if user_input == "h":
        print("type 'gain' to change alpha gain")
        print("type 'lift' to change alpha lift")
        print("type 'blur' to change alpha blur")
        print("type 'tl fg' to translate the foreground image")
        print("type 'tl bg' to translate the background image")
        print("type 'edge blur' or 'eb' to change the size of the edge blur")
        print("type 'light wrap' or 'lw' to change the size of the light wrap")
        print("type 'spill reduction' or 'sr' to activate/deactivate spill reduction")
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
    elif user_input == "edge blur" or user_input == "eb":
        eb_size = max(0, int(input("size of the edge blur: ")))
    elif user_input == "light wrap" or user_input == "lw":
        lw_size = max(0, int(input("size of the light wrap: ")))
        lw_intensity = max(0.0, float(input("intensity of the light wrap: ")))
    elif user_input == "spill reduction" or user_input == "sr":
        inp = input("spill reduction True/False: ")
        despill = inp == "True"

    alpha = create_matte()

    update_viewer(alpha)

    cv2.waitKey(1)
    user_input = input("Chance values (h: help, q: quit): ")


#TODO writing out comp image
#TODO writing out matte image
#TODO writing out foreground with alpha

# cleanup
cv2.destroyAllWindows()


#C:\Users\Jan\Pictures\3DProfile.jpg