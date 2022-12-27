import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
import os.path


class Painting():
    """
    Converting any image to a Painting By Numbers.
    The final product is the color segmented image, the outlined image for coloring and the palette.

    """

    def __init__(self, path_pic, nb_color=10, pixel_size=4000, save=True):
        self.namefile = os.path.basename(path_pic).split(".")[0]
        self.dirpath = os.path.dirname(path_pic)
        self.src = cv2.cvtColor(cv2.imread(path_pic), cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.save = save
        self.colormap = []

    def generate(self):
        """Main function for generating"""
        resized_img = self.resize()
        clean_img = self.cleaning(resized_img)
        segmented_image, colors = self.segmentation(clean_img)
        canvas = np.ones(segmented_image.shape[:3], dtype="uint8") * 255

        for ind, color in enumerate(colors):
            self.colormap.append([int(c) for c in color])
            mask = cv2.inRange(segmented_image, color, color)
            cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for contour in cnts:
                _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
                if width_ctr > 20 and height_ctr > 20 and cv2.contourArea(contour, True) < -100:
                    cv2.drawContours(canvas, [contour], -1, 0, 1)
                    # Add label
                    txt_x, txt_y = contour[0][0]
                    cv2.putText(canvas, '{:d}'.format(ind + 1),
                                (txt_x, txt_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

        palette = self.display_colormap()

        if self.save:
            cv2.imwrite(os.path.join(self.dirpath, f"{self.namefile}-colored.png"),
                        cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.dirpath, f"{self.namefile}-canvas.png"), cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(self.dirpath, f"{self.namefile}-palette.png"), cv2.cvtColor(palette, cv2.COLOR_BGR2RGB))

        return segmented_image, canvas, palette

    def resize(self):
        """Resize the image to match the target size and respect the picture ratio"""
        (height, width) = self.src.shape[:2]
        if height > width:
            dim = (int(width * self.tar_width / float(height)), self.tar_width)
        else:
            dim = (self.tar_width, int(height * self.tar_width / float(width)))
        return cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)

    def cleaning(self, picture):
        """Reduction of noize, morphological transformations """
        clean_pic = cv2.fastNlMeansDenoisingColored(picture, None, 10, 10, 7, 21)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
        clean_pic = cv2.morphologyEx(clean_pic, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
        return clean_pic

    def segmentation(self, picture):
        """Return the K-mean segmented image"""
        vectorized = np.float32(picture.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        attempts = 10
        ret, label, center = cv2.kmeans(vectorized, self.nb_color, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((picture.shape))
        return result_image, center

    def display_colormap(self):
        """Create palette as a picture for the user"""
        picture = np.ones((len(self.colormap) * 60 + 40, 600, 3), dtype="uint8") * 255
        for ind, col in enumerate(self.colormap):
            cv2.putText(picture, '{:d}'.format(ind + 1), (20, 60 * ind + 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
            cv2.rectangle(picture, (90, 60 * ind + 10), (170, 60 * ind + 50), col, thickness=-1)
            cv2.putText(picture, '{} {}'.format(self.rgb_to_name(col), col), (200, 60 * ind + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        return picture

    def rgb_to_name(self, rgb):
        """Convert RGB code to color name"""
        css3_db = CSS3_HEX_TO_NAMES
        names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))
        kdt_db = KDTree(rgb_values)
        distance, index = kdt_db.query(rgb)
        return names[index]


