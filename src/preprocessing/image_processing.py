import cv2
import matplotlib.pyplot as plt


def preprocess_image(image):
	plotter = Plotter()

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Enhance contrast
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced = clahe.apply(gray)
	plotter.add(enhanced, 'clahe')

	# Reduce noise
	blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
	plotter.add(blurred, 'blurred')

	# Binarize the image
	binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	plotter.add(binary, 'binary')

	# Apply morphological transformations
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
	plotter.add(closed, 'closed')

	# (Optional) Edge detection
	edges = cv2.Canny(closed, 50, 150)
	plotter.add(edges, 'edges')

	plotter.show()

	# cv2.imwrite('./preprocessing/output_7.jpg', blurred)


#'''
def preprocess_image(image):
	plotter = Plotter()

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	plotter.add(gray, "Grayscale")

	# contrast
	contrast = cv2.addWeighted(gray, 1.7, gray, 0, -140)
	plotter.add(contrast, "Output")

	thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	plotter.add(thresh, "Otsu")

	dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

	dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
	dist = (dist * 255).astype("uint8")
	plotter.add(dist, "Dist")

	dist = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	plotter.add(dist, "Dist Otsu")

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)
	plotter.add(opening, "Opening")

	final = cv2.bitwise_not(opening)
	plotter.add(final, "Final")

	# cv2.imwrite('./output.jpg', final)

	plotter.show()
#'''

class Plotter:
	def __init__(self):
		self.images = []
		self.descriptions = []

	def add(self, image, description: str):
		self.images.append(image)
		self.descriptions.append(description)

	def show(self):
		n = len(self.images)

		# Create a figure to display images
		fig, axs = plt.subplots(1, n, figsize=(50, 20))

		# If there is only one image, axs won't be a list, so we need to handle this case.
		if n == 1:
			axs = [axs]

		for i, (image, description) in enumerate(zip(self.images, self.descriptions)):
			# Read the image if a file path is given
			if isinstance(image, str):
				img = cv2.imread(image)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib
			else:
				img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Assume image is a valid OpenCV image object

			axs[i].imshow(img)
			axs[i].set_title(description)
			axs[i].axis('off')

		plt.show()

