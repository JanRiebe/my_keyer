import numpy as np
import cv2
import sys
from PySide2 import QtCore, QtWidgets

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


def cut2canvas(img, img_pos, canvas_size):
    # Creating an empty matrix to contain the cut off image.
    img_canvas = np.zeros([canvas_size[0], canvas_size[1], 3], 'float')

    # If the image lies outside the canvas we can return prematurely.
    if -img_pos[0] > img.shape[0] or -img_pos[1] > img.shape[1] \
            or img_pos[0] > canvas_size[0] or img_pos[1] > canvas_size[1]:
        return img_canvas

    # Finding whether the image overlaps the right side of the canvas.
    height_cut = min(img_pos[0] + img.shape[0], canvas_size[0])
    width_cut = min(img_pos[1] + img.shape[1], canvas_size[1])

    # Finding whether the image overlaps the left side of the canvas.
    pos_cut = [max(-img_pos[0], 0), max(-img_pos[1], 0)]

    # Cutting the image on the right and lower side.
    img_cut = img[pos_cut[0]: canvas_size[0] - img_pos[0], pos_cut[1]: canvas_size[1] - img_pos[1]]

    # Putting the cut image onto the canvas.
    img_canvas[max(0, img_pos[0]):height_cut, max(0, img_pos[1]):width_cut] = img_cut
    return img_canvas


def merge_images(foreground, matte, background):
    # spill reduction
    fg_despilled = foreground.copy()
    if despill:
        # if green bigger than average red and blue
        # lower green to average
        # increase r and b by a total of the amount g was reduced relative to each other
        b = fg_despilled[:, :, 0]
        g = fg_despilled[:, :, 1]
        r = fg_despilled[:, :, 2]
        average = (b + r) / 2           # the average between r and b
        g = np.minimum(g, average)      # lowering green values to the average of blue and red
        g_diff = g - average            # the amount green is being lowered
        b += g_diff / 2
        r += g_diff / 2
        fg_despilled[:, :, 0] = b
        fg_despilled[:, :, 1] = g
        fg_despilled[:, :, 2] = r

    # cutting foreground to view frustum
    fg_canvas = cut2canvas(fg_despilled, fg_pos, canvas)

    # cutting matte to view frustum
    matte_canvas = cut2canvas(matte, fg_pos, canvas)

    # cutting background to view frustum
    bg_canvas = cut2canvas(bg, bg_pos, canvas)

    # calculating comp image to be displayed
    comp = fg_canvas * matte_canvas + bg_canvas * (1 - matte_canvas)

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
        comp = comp * (1 - edge_img) + comp_blurred * edge_img

    # light wrap
    if lw_size > 0 and lw_intensity > 0:
        res, matte_binary = cv2.threshold(matte_canvas[:, :, 0], 0.5, 1.0, cv2.THRESH_BINARY_INV)
        kernel = np.ones((lw_size, lw_size), 'float')
        edge = cv2.dilate(matte_binary, kernel) - cv2.erode(matte_binary, kernel)
        edge = cv2.blur(edge, (lw_size, lw_size))
        edge *= (1 - matte_binary)
        light_wrap = np.zeros([canvas[0], canvas[1], 3], 'float')
        light_wrap[:, :, 0] = edge
        light_wrap[:, :, 1] = edge
        light_wrap[:, :, 2] = edge
        bg_blurred = cv2.blur(bg_canvas, (lw_size, lw_size))
        light_wrap *= bg_blurred
        comp = np.maximum(comp, light_wrap * lw_intensity)

    return comp, fg_canvas, matte_canvas


def update_viewer(foreground, alpha, background):
    """ draws the original image, the matte and the comp into a side by side viewer
    """

    viewer = np.ones([canvas[0], canvas[1], 3], 'float')

    if foreground is not None:
        # creating matte image to be displayed
        matte = np.ones([fg.shape[0], fg.shape[1], 3], 'float')
        matte[:, :, 0] = alpha
        matte[:, :, 1] = alpha
        matte[:, :, 2] = alpha

        if background is not None:
            comp, fg_canvas, matte_canvas = merge_images(foreground, matte, background)

            # show viewer
            viewer = np.concatenate((fg_canvas, matte_canvas, comp), axis=1)

        else:
            viewer = np.concatenate((fg, matte), axis=1)

    elif background is not None:
        viewer = background

    cv2.imshow("viewer", viewer)



def create_alpha():
    """ creates a matte from the foreground image
    :return: a matrix representing the alpha channel with values between 0.0 and 1.0
    """

    if fg is None:
        return None

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


# reading in images
def read_foreground(filepath):
    global fg
    fg = cv2.imread(filepath).astype(np.float32)/255.0
    return fg is not None


def read_background(filepath):
    global bg
    bg = cv2.imread(filepath).astype(np.float32)/255.0
    return bg is not None




#TODO writing out comp image
#TODO writing out matte image
#TODO writing out foreground with alpha




def update_gain(value):
    global a_gain
    a_gain = value


def update_lift(value):
    global a_lift
    a_lift = value


def update_matte_blur(value):
    global a_blur
    a_blur = max(0, int(value))


def update_edge_blur(value):
    global eb_size
    eb_size = max(0, int(value))


def update_light_wrap_size(value):
    global lw_size
    lw_size = max(0, int(value))


def update_light_wrap_intensity(value):
    global lw_intensity
    lw_intensity = max(0, value)


def toggle_spill_reduction(on):
    global despill
    despill = on
    refresh_view()


def update_fg_position_x(value):
    global fg_pos
    fg_pos[1] = int(value)


def update_fg_position_y(value):
    global fg_pos
    fg_pos[0] = int(value)


def update_bg_position_x(value):
    global bg_pos
    bg_pos[1] = int(value)


def update_bg_position_y(value):
    global bg_pos
    bg_pos[0] = int(value)


def refresh_view():
    alpha = create_alpha()
    update_viewer(fg, alpha, bg)


class FileLoader(QtWidgets.QHBoxLayout):
    def __init__(self, label_text, read_function):
        super(FileLoader, self).__init__()

        self.read_function = read_function

        self.label = QtWidgets.QLabel(label_text)
        self.inputFileLineEdit = QtWidgets.QLineEdit("enter path")
        self.fileDialogButton = QtWidgets.QPushButton("...")
        self.fileDialogButton.clicked.connect(self.on_input_file_button)

        self.addWidget(self.label)
        self.addWidget(self.inputFileLineEdit)
        self.addWidget(self.fileDialogButton)

    def on_input_file_button(self):
        filename, the_filter = QtWidgets.QFileDialog.getOpenFileName(self.fileDialogButton, ("Open Image"), ".", ("Image Files (*.png *.jpg *.bmp)"))
        if filename:
            self.inputFileLineEdit.setText(filename)
            self.read_function(filename)
            refresh_view()


class ValueSlider(QtWidgets.QHBoxLayout):
    def __init__(self, label_text, range_float, substeps, value_change_callback, release_callback):
        super(ValueSlider, self).__init__()

        self.value_change_callback = value_change_callback
        self.release_callback = release_callback
        self.substeps = substeps

        label = QtWidgets.QLabel(label_text)
        self.label_value = QtWidgets.QLabel("1")
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(range_float[0]*substeps, range_float[1]*substeps)
        slider.valueChanged.connect(self.value_changed)
        slider.sliderReleased.connect(release_callback)
        self.addWidget(label)
        self.addWidget(slider)
        self.addWidget(self.label_value)

    def value_changed(self, value):
        self.value_change_callback(value/self.substeps)
        self.label_value.setNum(value/self.substeps)


class LabeledCheckBox(QtWidgets.QHBoxLayout):
    def __init__(self, label_text, value_change_callback):
        super(LabeledCheckBox, self).__init__()

        label = QtWidgets.QLabel(label_text)
        check_box = QtWidgets.QCheckBox()
        check_box.setChecked(True)
        check_box.clicked.connect(value_change_callback)
        self.addWidget(label)
        self.addWidget(check_box)


class Controls(QtWidgets.QWidget):

    def __init__(self):
        super(Controls, self).__init__()

        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        layout.addLayout(FileLoader("foreground", read_foreground))
        layout.addLayout(FileLoader("background", read_background))
        layout.addLayout(ValueSlider("alpha gain", (1, 4), 1000, update_gain, refresh_view))
        layout.addLayout(ValueSlider("alpha lift", (-2, 0), 1000, update_lift, refresh_view))
        layout.addLayout(ValueSlider("alpha blur", (0, 20), 1, update_matte_blur, refresh_view))
        layout.addLayout(ValueSlider("edge blur", (0, 20), 1, update_edge_blur, refresh_view))
        layout.addLayout(ValueSlider("light wrap size", (0, 20), 1, update_light_wrap_size, refresh_view))
        layout.addLayout(ValueSlider("light wrap intensity", (0, 1), 1000, update_light_wrap_intensity, refresh_view))
        layout.addLayout(LabeledCheckBox("spill reduction", toggle_spill_reduction))
        layout.addLayout(ValueSlider("fg pos x", (-1000, 1000), 1000, update_fg_position_x, refresh_view))
        layout.addLayout(ValueSlider("fg pos y", (-1000, 1000), 1000, update_fg_position_y, refresh_view))
        layout.addLayout(ValueSlider("bg pos x", (-1000, 1000), 1000, update_bg_position_x, refresh_view))
        layout.addLayout(ValueSlider("bg pos y", (-1000, 1000), 1000, update_bg_position_y, refresh_view))

        self.setLayout(layout)

        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Controls')
        self.show()


app = QtWidgets.QApplication(sys.argv)
controls = Controls()
refresh_view()
sys.exit(app.exec_())


# cleanup
cv2.destroyAllWindows()
