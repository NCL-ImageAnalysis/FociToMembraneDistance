"""
<Authors> George Merces | Newcastle University | george.merces@newcastle.ac.uk
Dr James Grimshaw | Newcastle University | james.grimshaw@newcastle.ac.uk
Dr Calum Jukes | Newcastle University | calum.jukes@newcastle.ac.uk </Authors>
This Fiji macro runs in Jython and calculates the distance of spots 
from the membrane of rod shaped bacterial cells. It requires a trained Ilastik model 
and a folder within the selected home folder called "Raw_Images" containg nd2 files
"""


#@ File (label="Home Folder:", style="directory") Home_Folder
#@ File (label="Ilastik.exe file:", style="file") IlastikExe
#@ Float (label="Foci Diameter (um):", style="format:#####.#####") Foci_Diameter
#@ Float (label="Trackmate Quality Threshold:", style="format:#####.#####") TrackMate_Quality
#@ Float (label="Green to Red Chromatic Abberation Scaling Factor:", style="format:#####.#####") RedChromAbCorr
#@ Float (label="Green to UV Chromatic Abberation Scaling Factor:", style="format:#####.#####") UVChromAbCorr
#@ String (label="Membrane Channel Colour:", choices={"Green", "Red"}) Membrane_Channel_Colour
#@ Integer (label="Membrane Channel:") Memb_Chan
#@ Integer (label="Foci Channel:") Foci_Chan
#@ Integer (label="Nucleoid Channel:") Nuc_Chan

import os, re, sys
from java.lang import Math, Double
from java.io import File
from ij import IJ
from ij.io import FileSaver
from ij.gui import NonBlockingGenericDialog, Roi, Line, PointRoi, OvalRoi, ProfilePlot
from ij.measure import Measurements, ResultsTable, CurveFitter
from ij.plugin import Scaler, RGBStackMerge, RoiEnlarger
from ij.plugin.frame import RoiManager 
from ij.plugin.filter import Analyzer
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from org.scijava import Context
from org.scijava.app import DefaultStatusService
from net.imglib2.img import ImagePlusAdapter
from org.scijava.log import StderrLogService
from org.scijava.service import ServiceHelper
from org.ilastik.ilastik4ij.executors import PixelClassification
from net.imglib2.img.display.imagej import ImageJFunctions
from fiji.plugin.trackmate import Model, Settings, SelectionModel, TrackMate
from fiji.plugin.trackmate.detection import LogDetectorFactory, LogDetector
from fiji.plugin.trackmate.features import FeatureFilter
from net.imglib2.img.display.imagej import ImageJFunctions
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.visualization.hyperstack import HyperStackDisplayer
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory

# Settings----------------------------------------------------------------v
# Ilastik Settings---------------------------------------------------v
# Output that is used for segmentation
IlastikOutputType = "Probabilities"
# Location of ilastik.exe file
# IlastikExe = r'C:\New_folder\ilastik-1.4.0rc8\ilastik.exe' # Home Computer
# IlastikExe = r'C:\Other_Program_Files\ilastik-1.3.3post3\ilastik.exe' # CBCB Computer
# IlastikExe = r'C:\ilastik-1.4.0-gpu\ilastik.exe' #Image analysis computer
# Number of threads ilastik is allowed to use (-ve is no restriction)
Threads = -1
# Amount of ram Ilastik is allowed to use
IlRam = 4096
#--------------------------------------------------------------------^

# Segmentation settings--------------------------------------------------v
# Sigma for gaussian blur. If not desired set to None
Gaussian_BlurSigma = 2
# ImageJ thresholding method for segmenting membrane probability
ThresholdingMethod = "Minimum"
# Number of pixels you want to dilate the cells by following segmentation
DilationPixels = 3
# Whether you want to apply a watershed process to the binary
WatershedBool = False
#------------------------------------------------------------------------^
# Analyze particles settings-v
Cell_Sizes = "200-inf"
Cell_Circularity = "0.00-1.00"
Roi_Enlargement = 3
#----------------------------^

# TrackMate settings---------------v
# Image scale in microns per pixel
PixelScale = 0.065
#----------------------------------^

# Membrane width measurement settings-v
# R^2 value that gausian fits must exceed
R2_Cutoff = 0.95
# Total number of pixels you want to offset FWHM by to match membrane
# Positive values put membrane outside of FWHM, negative brings membrane in
FWHM_Modifier = -4
# Range of angles you want to search to find minimum membrane width
Angle_Range = 20
#-------------------------------------^

#-------------------------------------------------------------------------^

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
#--------------------------------------------FUNCTIONS--------------------------------------------#

def checkInputs(HomePath):
	"""Takes ij.gui.GenericDialog and returns a list with processed inputs
	
	List contains: 
	[Home folder (str), 
	Prominence setting (str), 
	Chromatic abberation correction (float), 
	Membrane channel number (int), 
	Foci channel number (int), 
	Membrane channel colour (str), 
	Foci channel colour (str), 
	Ilastik exe file (Java.io.File), 
	Ilastik project file (Java.io.File), 
	Path to raw image folder (str),
	List of .nd2 files ([str])]"""


	# Checks for and processes the ilastik.exe file input------v
	# Checks that the path to the ilastik.exe file is correct
	if IlastikExe.exists() == False:
		IJ.error(
			"No ilastik.exe file found in given location. \
			Please input into macro settings"
		)
		return False
	
	# Gets the ilastik project file from the home folder and---------------v
	# returns it as a Java file object
	# Gets all the files in the home folder
	HomeFiles = os.listdir(HomePath)
	# Gets a list of files that have the 
	# .ilp extension to find Ilastik projects
	ilp_re_Obj = re.compile(r'\.ilp$', flags=re.IGNORECASE)
	IlastikProjectFiles = filter(ilp_re_Obj.search, HomeFiles)
	# Checks that exactly one ilastik project is present
	if len(IlastikProjectFiles) != 1:
		if len(IlastikProjectFiles) > 1:
			IJ.error("Multiple Ilastik project files detected")
		else:
			IJ.error("No Ilastik project file detected")
		return False
	# Converts the Ilastik project file into a Java File object
	ProjectFilePath = os.path.join(HomePath, IlastikProjectFiles[0])
	IlastikProjectFile = File(ProjectFilePath)
	# Adds the project file to the list that is returned

	#----------------------------------------------------------------------^

	# Checks that the channel numbers are not the same---v
	if (Memb_Chan == Foci_Chan or
	Memb_Chan == Nuc_Chan or
	Foci_Chan == Nuc_Chan):
		IJ.error("Selected channel colours cannot match")
		return False
	#----------------------------------------------------^

	# Checks for the raw image folder and returns it-------------------------v
	# Generates the path to the folder containing the raw images
	RawImagePath = os.path.join(HomePath, "Raw_Images")
	# Checks that a home folder was chosen 
	# that has a subfolder named "Raw_Images"
	if os.path.exists(RawImagePath) == False:
		IJ.error(
			"Please choose a home folder containing a 'Raw_Images' subfolder"
		)
		return False
	#------------------------------------------------------------------------^

	# Gets a list of images, filters for .nd2 files and returns as a list-v
	# Gets a list of filenames of the raw images
	RawImageList = os.listdir(RawImagePath)
	# This removes any filenames that do not have a .nd2 extension
	nd2_re_Obj = re.compile(r'\.nd2$', flags=re.IGNORECASE)
	ImageList = filter(nd2_re_Obj.search, RawImageList)
	# Checks that one or more images are present in the folder
	if len(ImageList) == 0:
		IJ.error("No nd2 files were detected")
		return False
	#---------------------------------------------------------------------^
	return IlastikProjectFile, RawImagePath, ImageList

def scaleAndCrop(Scale_This_ImagePlus, Scaling_Factor):
	"""Takes ij.ImagePlus, scaling factor as a float and returns a 
	scaled ij.ImagePlus which has been cropped down to given 
	width/height to correct for chromatic abberation between channels"""
	
	# Gets the dimensions of the image, only Width and Height are needed
	(Width, 
	Height, 
	nChannels, 
	nSlices, 
	nFrames) = Scale_This_ImagePlus.getDimensions()
	# Gets the scaled width and height. 
	# Need to use java rounding to keep consistent
	NewWidth = Math.round(Width * Scaling_Factor)
	NewHeight = Math.round(Height * Scaling_Factor)
	# Runs the scale command with options being: 
	# Image, Width, Height, Depth, interpolation option
	ScaledImage = Scaler.resize(
		Scale_This_ImagePlus, 
		NewWidth, 
		NewHeight, 
		1, 
		"bilinear"
	)
	# Gets the distance in pixels from the top left corner 
	# that the ROI needs for setting in the centre
	Roi_X_Offset = Math.round((NewWidth-Width)/2)
	Roi_Y_Offset = Math.round((NewHeight-Height)/2)
	# Creates an Roi that will be used to crop the image down 
	# so they are the same size. Settings are x, y, width, height
	CropRoi = Roi(Roi_X_Offset, Roi_Y_Offset, Width, Height)
	# Applies the roi to the image
	ScaledImage.setRoi(CropRoi)
	# Crops the image to size
	CroppedImage = ScaledImage.crop()
	# Closes the upscaled image to conserve memory
	ScaledImage.close()
	# Returns the cropped image to the main body of the code
	return CroppedImage

def chromaticAbberationCorrection(
		ArrayofImages, 
		RedChromaticAbberationSetting,
		UVChromaticAbberationSetting, 
		MembraneChannelNumber, 
		FociChannelNumber,
		NucleoidChannelNumber, 
		MembraneChannelColour,
		BaseFilepath):
	"""Corrects for chromatic abberation between 
	red and green channels and saves the corrected images
	
	Takes images from bioformats import with split 
	channels enabled [ImagePlus], scaling factor 
	for chromatic abberation (float), 
	Channel number for membrane (int), and foci (int), 
	channel colour of the membrane (str), filepath to save folder
	  with an extensionless filename (str) and returns the corrected membrane 
	  and foci channels as ij.ImagePlus objects"""

	# Depending on which colour is membrane and which colour is foci 
	# will send the appropriate channel to have the 
	# chromatic abberation correction
	if MembraneChannelColour == "Red":
		Membrane_Image = scaleAndCrop(
			ArrayofImages[MembraneChannelNumber], 
			RedChromaticAbberationSetting
		)
		Foci_Image = ArrayofImages[FociChannelNumber]
	else:
		Foci_Image = scaleAndCrop(
			ArrayofImages[FociChannelNumber], 
			RedChromaticAbberationSetting
		)
		Membrane_Image = ArrayofImages[MembraneChannelNumber]
	Nucleoid_Image = scaleAndCrop(
		ArrayofImages[NucleoidChannelNumber], 
		UVChromaticAbberationSetting
	)
	Foci_Image.resetDisplayRange()
	Membrane_Image.resetDisplayRange()
	Nucleoid_Image.resetDisplayRange()
	# Merges the channels so the file will be 
	# saved after correction for chomatic abberation
	ChromCorrTiff = RGBStackMerge.mergeChannels(
		[Foci_Image, Membrane_Image, Nucleoid_Image], 
		True
	)
	# Saves the corrected membrane image in the Chr_Abberation_Corrected folder
	FileSaver(Membrane_Image).saveAsTiff(BaseFilepath + "_Membrane.tiff")
	# Saves the corrected foci image in the Chr_Abberation_Corrected folder
	FileSaver(Foci_Image).saveAsTiff(BaseFilepath + "_Foci.tiff")
	# Saves the corrected Nucleoid image in the Chr_Abberation_Corrected folder
	FileSaver(Nucleoid_Image).saveAsTiff(BaseFilepath + "_Nucleoid.tiff")
	# Saves the corrected merged image in the Chr_Abberation_Corrected folder
	FileSaver(ChromCorrTiff).saveAsTiff(BaseFilepath + "_Merged.tiff")
	return Membrane_Image, Foci_Image, Nucleoid_Image

def generateMembraneProbability(
		MembraneImagePlus, 
		Classification_Obj, 
		Classification_Type, 
		BaseFilename, 
		BaseFolderpath):
	"""Runs the ilastik model on the membrane channel 
	and saves/returns the output
	
	Takes Membrane channel (ij.ImagePlus), 
	org.ilastik.ilastik4ij.executors.PixelClassification, 
	PixelClassification.PixelPredictionType, 
	extensionless filename (str), 
	path to folder where probability map will be saved (str) 
	and returns the probability map (ij.ImagePlus)"""

	# Need to convert this from ImagePlus -> ImgPlus
	Membrane_ImgPlus = ImagePlusAdapter.wrapImgPlus(MembraneImagePlus)
	# Runs the actual classify pixels command in ilastik
	IlastikImgPlus = Classification_Obj.classifyPixels(
		Membrane_ImgPlus, 
		Classification_Type
	)
	# Generates the filename for the membrane probability image 
	ProbFilename = BaseFilename + "_Membrane-Probability.tiff"
	# Converts the file into an ImagePlus for saving
	Ilastik_ImagePlus = ImageJFunctions.wrap(IlastikImgPlus, ProbFilename)
	# Saves the probability image based upon the membrane channel
	FilePath = os.path.join(BaseFolderpath, ProbFilename)
	FileSaver(Ilastik_ImagePlus).saveAsTiff(FilePath)
	return Ilastik_ImagePlus

def segmentIlastikOutput(
		Ilastik_ImagePlus, 
		Gaussian_Setting, 
		Thesholding_Method, 
		Dilation_Setting, 
		Watershed_Bool, 
		BaseFilepath):
	"""Segements the probability map generated by 
	ilastik and saves/returns the binary image
	
	Takes probability map (ij.ImagePlus), Gaussian sigma (int/bool), 
	Thresholding method (str), number of pixels to dilate the binary (int), 
	bool for whether to apply a watershed, filepath to save folder with an 
	extensionless filename (str) and returns the binary image (ij.ImagePlus)"""

	# If gaussian blur is enabled will perform it 
	# with the sigma value given in macro settings
	if type(Gaussian_Setting) == int: 
		IJ.run(
			Ilastik_ImagePlus, 
			"Gaussian Blur...", 
			"sigma=" + str(Gaussian_Setting)
		)

	# Sets the threshold based upon the thesholding method in the settings
	IJ.setAutoThreshold(Ilastik_ImagePlus, Thesholding_Method + " dark")
	# Applys the thresholding to get a binary image
	IJ.run(Ilastik_ImagePlus, "Convert to Mask", "")
	# This loop will run the dilate command as many times 
	# as the number of dilation pixels specified in settings
	IterDil = 0
	while IterDil < Dilation_Setting:
		IJ.run(Ilastik_ImagePlus, "Dilate", "")
		IterDil += 1
	# Will run the watershed processing step if enabled in the settings
	if Watershed_Bool == True:
		IJ.run(Ilastik_ImagePlus, "Watershed", "")
	# Saves the segmented membrane image
	FileSaver(Ilastik_ImagePlus).saveAsTiff(
		BaseFilepath 
		+ "_Semented-Membrane.tiff"
	)
	return Ilastik_ImagePlus


def getCellRoi(
		Binary_Image, 
		Cell_Size_Setting, 
		Cell_Circularity_Setting, 
		File_Path):
	"""Runs the analyze particles command on a binary image 
	and saves/returns the resulting rois and a combined roi
	
	Takes Binary image (ij.ImagePlus), Min/Max size (str), 
	Min/Max circularity (str), file path to save the rois"""

	# Runs the analyze particles command to get cell ROI. 
	# Done by adding to the overlay in order to not have ROIManger shown to user
	AnalyzeParticlesSettings = (
		"Size=" 
		+ Cell_Size_Setting 
		+ " circularity=" 
		+ Cell_Circularity_Setting 
		+ " clear overlay exclude"
	)

	IJ.run(Binary_Image, "Analyze Particles...", AnalyzeParticlesSettings)
	# Gets the Overlayed ROIs from analyze particles
	Overlayed_Rois = Binary_Image.getOverlay()
	# Takes the overlay and turns it into an array of ROI
	CellRoiList = Overlayed_Rois.toArray()
	# Removes this overlay to clean up the image
	IJ.run(Binary_Image, "Remove Overlay", "")
	Roi_Manger = RoiManager(True)
	# Adds all the ROI to the ROI Manager
	for CellNumber in range(0, len(CellRoiList)):
		# Renames the roi to give it a name 
		# to trace data back to individual cells
		CellRoiList[CellNumber].setName("Cell_" + str(CellNumber + 1))
		Roi_Manger.addRoi(CellRoiList[CellNumber])
	# Deselects any selected ROI so whole list of ROI is saved
	Roi_Manger.deselect()
	# Saves the cell ROIs
	Roi_Manger.save(File_Path)
	# Combines the cell ROIs
	Roi_Manger.runCommand(Binary_Image, "Combine")
	# Gets the combined ROIs from the image
	MergedRois = Binary_Image.getRoi()
	# Deselects the ROI from the image
	MembraneBinary.resetRoi()
	# Clears the RoiManager
	Roi_Manger.close()
	return CellRoiList, MergedRois

def getRoiMeasurements(SampleRoi, Image, Measurement_Options):
	"""Gets the mean value of the provided Roi for the given image
	
	Takes an ij.gui.Roi and an ij.ImagePlus and returns a double"""
	# Initialises a new empty results table
	RTable = ResultsTable()
	# Initialises an Analyzer object using 
	# the image and the empty results table
	An = Analyzer(Image, RTable)
	Image.setRoi(SampleRoi)
	An.measure()
	OutputList = []
	for Option in Measurement_Options:
		OutputList.append(RTable.getValue(Option, 0))
	RTable.reset()
	Image.resetRoi()
	return OutputList

def getBackground(Image, MergedRois, PixelIncrease):
	"""Deletes the pixels no found within a certain distance of segmented cells
	
	Takes image to be subtracted (ij.ImagePlus), 
	Merged cell rois (ij.gui.Roi), 
	Number of pixels to enlarge rois by (int), 
	file path where subtracted image will be saved (str)"""

	# Enlarges the rois before inversion so foci 
	# just outside the membrane are not recognised
	enlarged_rois = RoiEnlarger().enlarge(MergedRois, PixelIncrease)
	# Assigns the ROI to the foci image
	Image.setRoi(enlarged_rois)
	# Inverts the cell roi so selecting the background
	IJ.run(Image, "Make Inverse", "")
	InvertedRoi = Image.getRoi()
	# Gets the mean value of the background
	BackgroundMean = getRoiMeasurements(InvertedRoi, Image, ["Mean"])[0]
	Image.resetRoi()
	return BackgroundMean

def getTrackmateFoci(Image, Radius, QualityThresh):
	# Removing the scale so that can more easily convert to ROIs
	IJ.run(Image, "Set Scale...", "distance=0 known=0 unit=pixel")
	Image.resetRoi()
	# Creates a TrackMate model object where data is saved
	TModel = Model()
	# Creates a TrackMate settings object and loading the image
	TSettings = Settings(Image)
	# Using Laplacian of Gaussian (LoG) detector
	TSettings.detectorFactory = LogDetectorFactory()
	TSettings.detectorSettings = {
		'DO_SUBPIXEL_LOCALIZATION' : True,
		'RADIUS' : Radius,
		'TARGET_CHANNEL' : 1,
		'THRESHOLD' : 0.,
		'DO_MEDIAN_FILTERING' : False
		}
	# Adding the quality filter
	QFilter = FeatureFilter('QUALITY', QualityThresh, True)
	TSettings.addSpotFilter(QFilter)
	# Initialising the TrackMate object
	TMate = TrackMate(TModel, TSettings)
	# Runs the detection
	TMate.execDetection()
	# Gets rid of very low quality spots
	TMate.execInitialSpotFiltering()
	# Filters based upon given quality threshold
	TMate.execSpotFiltering(False)
	# Dictionary of the quality of each spot
	QualityDict = {}
	# Dictionary of the spot rois
	SpotDict = {}
	# Gets the collection of spots from the model
	SpotCollection = TModel.getSpots()
	# Iterates through each spot in the collection
	for Spot in SpotCollection.iterable(True):
		# Gets X and Y coordinates of the spot
		Spot_X = Spot.getFeature('POSITION_X')
		Spot_Y = Spot.getFeature('POSITION_Y')
		# Gets the quality of the spot
		Quality = Spot.getFeature('QUALITY')

		# Oval Rois need to be defined by the top left corner so
		# coords have to be offset by the radius and half a pixel
		Offset = Radius - 0.5
		OffsetPixelX = Spot_X - Offset
		OffsetPixelY = Spot_Y - Offset
		# Creates an OvalRoi equivalent to the trackmate SpotRoi
		OvRoi = OvalRoi(
			OffsetPixelX, 
			OffsetPixelY, 
			Radius*2, 
			Radius*2
		)
		# Foci will be stored as a tuple 
		# of the X and Y coordinates
		Foci = (Spot_X, Spot_Y)
		# Adds the quality and roi to their respective
		# dictionaries with foci tuple as the key
		QualityDict[Foci] = Quality
		SpotDict[Foci] = OvRoi
	return QualityDict, SpotDict

def rasteriseOutline(x_array, y_array):
	"""Takes pair of integer lists corresponding to 
	x/y coordinates of a polygon and returns 
	pair of integer lists corresponding to 
	coordinates for the whole outline of the polygon"""

	# Defines arrays that will be filled with connected coordinates
	connected_x = []
	connected_y = []
	# Iterates through each coordinate in the polygon
	for coordIndex in range(0, len(x_array)):
		# Checks whether the x coordinate is going up or down
		if x_array[coordIndex] - x_array[coordIndex - 1] > 0:
			x_incrementer = 1
		else:
			x_incrementer = -1
		try:
			# Gets the slope of the line between the two polygon points
			slope = (
				float(y_array[coordIndex] - y_array[coordIndex - 1])
				/float(x_array[coordIndex] - (x_array[coordIndex - 1]))
			)
			# Calculates the intercept of this slope 
			# which is needed for calculating the final value
			Intercept = y_array[coordIndex] - (slope * x_array[coordIndex])
			# Variable for increasing/decreasing value of x
			Incrementing_X = x_array[coordIndex - 1]
			# This loop will run untill the two points given connect
			while Incrementing_X != x_array[coordIndex]:
				# Increases/decreases value of incrementer
				Incrementing_X += x_incrementer
				# Calculates the new Y value
				Calculated_Value = (Incrementing_X * slope) + Intercept
				# Rounds Y values, converting it into an integer
				Rounded_Y = Math.round(Calculated_Value)
				# Appends both values to their respective lists
				connected_x.append(Incrementing_X)
				connected_y.append(Rounded_Y)
		# If two x coordinates are equal then will get a zero 
		# division error so this calculation is instead used
		except(ZeroDivisionError):
			# Checks whether the y coordinate is going up or down
			if y_array[coordIndex] - y_array[coordIndex - 1] > 0:
				y_incrementer = 1
			else:
				y_incrementer = -1
			# Variable for increasing/decreasing value of y
			Incrementing_Y = y_array[coordIndex - 1]
			# This loop will run untill the two points given connect
			while Incrementing_Y != y_array[coordIndex]:
				# Increases/decreases value of incrementer
				Incrementing_Y += y_incrementer
				# Appends both values to their 
				# respective lists (x value will stay the same)
				connected_x.append(x_array[coordIndex])
				connected_y.append(Incrementing_Y)
		return connected_x, connected_y

def distanceBetweenPoints(X1, Y1, X2, Y2):
	xdiff = X1 - X2
	ydiff = Y1 - Y2
	Distance = Math.sqrt((xdiff*xdiff) + (ydiff*ydiff))
	return Distance


def nearestNeighbourPoints(PolyRoi, MarkerPoint):
	"""Takes ij.gui.PolygonRoi and ij.gui.PointRoi and returns the 
	closest distance between the PointRoi and the outer edge 
	of the PolygonRoi as a float"""

	# Sets the initial distance to infinity 
	# so will always be lower than one value
	distance = float('inf')
	# Gets the outline as a polygon
	Poly = PolyRoi.getPolygon()
	# Calls the rasteriseOutline function to connect points in polygon
	x_arr, y_arr = rasteriseOutline(Poly.xpoints, Poly.ypoints)
	# This loop determines the distance of each 
	# coordinate in the polygon from the given point
	for arrayindex in range(0, len(x_arr)):
		tempdist = distanceBetweenPoints(
			MarkerPoint[0], 
			MarkerPoint[1], 
			x_arr[arrayindex], 
			y_arr[arrayindex]
		)
		if tempdist < distance:
			distance = tempdist
	return distance

def checkSoleContainer(
		Point, 
		enlarged_CellRoi, 
		CellRoi, 
		Cell_to_Points_Dict, 
		Points_to_Cell_Dict):
	"""Checks if the given foci is solely contained by/closest to the given cell roi
	
	Takes foci (java.awt.Point), enlarged cell roi (ij.gui.Roi), 
	cell roi (ij.gui.Roi), Cell_to_Points_Dict (dict), 
	Points_to_Cell_Dict (dict)"""

	# Checks if the point falls within the enlarged roi
	if enlarged_CellRoi.containsPoint(Point[0], Point[1]):
		# Checks if the point has already been assigned to another cell
		if Point not in Points_to_Cell_Dict:
			return True, Cell_to_Points_Dict
		# Gets the other cells roi from the 
		# dictionary with the point as the key
		OtherCellRoi = Points_to_Cell_Dict[Point]
		# Checks if the point is within the un-enlarged (current) roi
		if CellRoi.containsPoint(Point[0], Point[1]):
			# Removes the point from being assigned to the old cell roi
			Cell_to_Points_Dict[OtherCellRoi].remove(Point)
			return True, Cell_to_Points_Dict
		# Checks if instead the point is within the un-enlarged (old) roi
		elif OtherCellRoi.containsPoint(Point[0], Point[1]):
			# If true then no dictionaries need altering
			return False, Cell_to_Points_Dict
		# If it falls within neither roi then will 
		# determine the closest roi to it
		# Calls the nearestNeighbourPoints function to 
		# get the closest distance to one of the points
		CellDistance = nearestNeighbourPoints(CellRoi, Point)
		OtherCellDistance = nearestNeighbourPoints(OtherCellRoi, Point)
		# Checks if the new roi is closest
		if CellDistance < OtherCellDistance:
			# Removes the point from being assigned to the old cell roi
			Cell_to_Points_Dict[OtherCellRoi].remove(Point)
			return True, Cell_to_Points_Dict
	else:
		return False, Cell_to_Points_Dict

def getFoci(Image, Radius, Quality, CellRoiList, PixelIncrease):
	"""Gets the foci found within each cell roi
	
	Takes the foci image (ij.ImagePlus), Prominence setting (str), 
	array of cell rois ([ij.gui.Roi]), 
	Number of pixels to enlarge rois by (int) 
	and returns a dictionary with cell rois as keys (ij.gui.Roi) 
	and list of points as values ([java.awt.Point]), 
	an roi containing all foci (ij.gui.PointRoi)"""

	QualityDict, SpotDict = getTrackmateFoci(Image, Radius, Quality)
	# Creates dictionaries to get per cell points
	Cell_to_Points_Dict = {}
	Points_to_Cell_Dict = {}
	# This loop will go through each roi and determine which 
	# points are within an enlarged form of it
	# If the point falls into the expanded area of multiple foci. 
	# It will be associated with the closest cell 
	for CellRoi in CellRoiList:
		# Enlarges the roi so foci just outside the membrane are recognised
		enlarged_CellRoi = RoiEnlarger().enlarge(CellRoi, PixelIncrease)
		# Iterates through all of the points for every cell
		for Point in QualityDict:
			# Checks whether the point should be assigned to this cell
			SaveBool, Cell_to_Points_Dict = checkSoleContainer(
				Point, 
				enlarged_CellRoi, 
				CellRoi, 
				Cell_to_Points_Dict, 
				Points_to_Cell_Dict
			)

			if SaveBool:
				# Adds the roi to the dictionary using points as keys
				Points_to_Cell_Dict[Point] = CellRoi
				# Try except will speed up this process
				try:
					# Appends this point to the 
					# dictionary using the roi as a key
					Cell_to_Points_Dict[CellRoi].append(Point)
				except(KeyError):
					# Adds a list containing this point to 
					# the dictionary using the roi as a key
					Cell_to_Points_Dict[CellRoi] = [Point]
	return Cell_to_Points_Dict, QualityDict, SpotDict

def saveFociRoi(CellRoiList, Cell_to_Points_Dict, File_Path):
	"""Saves the foci in the provided dictionary as roi
	
	Takes array of cell rois ([ij.gui.Roi]), a dictionary with 
	cell rois as keys and list of their containing foci as 
	points the values ({ij.gui.Roi = [java.awt.Point]}), 
	an roi containing all foci (ij.gui.PointRoi), 
	file path to save the roi (str)"""

	# Creates an RoiManager Instance
	Roi_Manger = RoiManager(True)
	# Uses the list so can maintain this order going forward
	for AnalysedRoi in CellRoiList:
		# Sets the name of the roi to match the cell roi name
		CellRoiName = AnalysedRoi.getName()
		# If the cell has no foci then will not save the PointRoi
		try:
			FociNumber = 0
			# Will go through the foci found in the list entry 
			# in the dictionary and add them to a single PointRoi
			for CellFoci in Cell_to_Points_Dict[AnalysedRoi]:
				# Creates an empty PointRoi obj
				SaveRoi = PointRoi(CellFoci[0], CellFoci[1])
				SaveRoi.setName(CellRoiName + "-Foci_" + str(FociNumber + 1))
				# Adds the point to the roi manager
				Roi_Manger.addRoi(SaveRoi)
				FociNumber += 1
		# If the cell has no foci then will not save the PointRoi
		except(KeyError):
			pass
	# Saves the roi as a zip file using the image filename 
	Roi_Manger.save(File_Path)
	Roi_Manger.close()

def getCellMeasurements(RoiList, Image, Background):
	OutputDict = {}
	for CellRoi in RoiList:
		OutputDict[CellRoi] = getRoiMeasurements(
			CellRoi, 
			Image, 
			["Mean", "Area", "Major", "Max"]
		)
		# If the roi contains overexposed pixels 
		# will set mean value to infinity
		if OutputDict[CellRoi][3] == 65535:
			OutputDict[CellRoi][0] = float("inf")
		# Subtracks the background from the mean value
		OutputDict[CellRoi][0] -= Background
	return OutputDict

def getLineEndCoords(X, Y, Length, Angle):
	"""Takes the x and y the starting point of the line, 
	the desired length and angle of a line and returns 
	the end coordinates of that line"""

	# Need to convert the angle to radians 
	Radian = Math.toRadians(Angle)
	# These equations generate the end coordinates and returns them
	End_X = X + (Length * Math.cos(Radian))
	End_Y = abs(-Y + (Length * Math.sin(Radian)))
	return End_X, End_Y


def fullWidthHalfMax(Image, LineRoi):
	"""Gets the fluorescence intesnsity profile of a line, 
	fits a gaussian curve and calculates the 
	full width half maximum of that curve
	
	Takes an Image (ij.ImagePlus) and a Line roi (ij.gui.Line) 
	and returns the parameters: minimum, maximum, 
	centre of the peak, standard deviation, R^2 ([double]) 
	and the full width half maximum of the gaussian curve (double)"""
	
	# Adds the line roi to the image
	Image.setRoi(LineRoi)
	# Generates the fluorescence intesnsity profile
	Fluor_Profile = ProfilePlot(Image)
	# Gets the Plot obj 
	Fluor_Plot = Fluor_Profile.getPlot()
	# Pulls the values for X from the plot (as float array)
	X_Values = Fluor_Plot.getXValues()
	# Pulls the values for Y straight from 
	# the ProfilePlot (as a double array)
	Y_Values = Fluor_Profile.getProfile()
	# Converts the float array of X values into a 
	# double array (needed for the curve fitting)
	X_Double = []
	for Value in X_Values:
		X_Double.append(Double(Value))
	# Fits the gaussian curve
	Fit = CurveFitter(X_Double, Y_Values)
	Fit.doFit(Fit.GAUSSIAN)
	# Gets the parameters from the fit as a double array
	# Parameters are: minimum, maximum, centre of the peak, 
	# standard deviation and sum of residuals squared
	Parameters = Fit.getParams()
	# Gets the R^2 value
	R_Squared = Fit.getRSquared()
	# Gets the standard deviation of residuals
	SDevRes = Fit.getSD()
	# Calculates the full width half max from 
	# the standard deviation parameter
	FullWidHMax = Parameters[3] * (2 * Math.sqrt(2 * Math.log(2)))
	# Clears the roi from the image
	Image.resetRoi()
	return Parameters, FullWidHMax, R_Squared, SDevRes

def distanceFromMembrane(Angle, Foci, Image, Length):
	"""Gets the width of the membrane and distance of foci from 
	membrane edges and centre of the cell"""

	# Need to make sure when getting the other side 
	# of the line the angle isnt going over 360 degrees 
	if Angle > 180:
		ModdedAngle = Angle - 180
	else:
		ModdedAngle = Angle + 180
	# Gets one side of the extended line 
	Start_X, Start_Y = getLineEndCoords(Foci[0], Foci[1], Length, Angle)
	# Gets the other side of the extended line
	End_X, End_Y = getLineEndCoords(Foci[0], Foci[1], Length, ModdedAngle)
	# Generates the line roi
	TempLineRoi = Line(Start_X, Start_Y, End_X, End_Y)
	# Sets the width to 3 pixels wide so there is some 
	# averaging on the fluorescence intensity profile
	TempLineRoi.setStrokeWidth(3)
	# Gets the parameters of a gaussian fit (minimum, maximum, 
	# centre of the peak, standard deviation, sum of residuals squared),
	# the full width half maximum, the R^2 value and 
	# the standard deviation of residuals
	Parameters, FullWidHMax, R_Squared, SDevRes = fullWidthHalfMax(
		Image, 
		TempLineRoi
	)
	ModifiedFWHM = FullWidHMax + FWHM_Modifier
	# Finds the distance from the start of the line 
	# to the start of the membrane
	DistanceToMembraneStart = Parameters[2] - (ModifiedFWHM/2)
	# Gets the coordinates of the start of the membrane
	Membrane_Start_X, Membrane_Start_Y = getLineEndCoords(
		Start_X, 
		Start_Y, 
		DistanceToMembraneStart, 
		ModdedAngle
	)
	# Gets the coordinates of the centre peak of the curve
	Membrane_Centre_X, Membrane_Centre_Y = getLineEndCoords(
		Membrane_Start_X, 
		Membrane_Start_Y, 
		ModifiedFWHM/2, 
		ModdedAngle
	)
	# Gets the coordinates of the end of the membrane
	Membrane_End_X, Membrane_End_Y = getLineEndCoords(
		Membrane_Start_X, 
		Membrane_Start_Y, 
		ModifiedFWHM, 
		ModdedAngle
	)
	# Generates the line roi for this  membrane  spanning line
	MembraneLineRoi = Line(
		Membrane_Start_X, 
		Membrane_Start_Y, 
		Membrane_End_X, 
		Membrane_End_Y
	)
	# Sets the width of this line to 2 pixels so can 
	# check if it contains the foci properly
	MembraneLineRoi.setStrokeWidth(2)
	# Distance of the foci from the start of the membrane
	startdist = distanceBetweenPoints(
		Membrane_Start_X, 
		Membrane_Start_Y, 
		Foci[0], 
		Foci[1]
	)
	# Distance of the foci from the centre of the cell
	centredist = distanceBetweenPoints(
		Membrane_Centre_X, 
		Membrane_Centre_Y, 
		Foci[0], 
		Foci[1]
	)
	# Distance of the foci from the end of the membrane 
	enddist = distanceBetweenPoints(
		Membrane_End_X, 
		Membrane_End_Y, 
		Foci[0], 
		Foci[1]
	)
	MembraneAreaRoi = MembraneLineRoi.convertLineToArea(MembraneLineRoi)
	# Checks whether the line has the shortest distance found so far, 
	# if the R^2 of the fit is greater than a certain thresholding 
	# and that it contains the foci
	if R_Squared > R2_Cutoff and MembraneAreaRoi.containsPoint(Foci[0], Foci[1]):
		# Finds which distance is shorter the distance from the 
		if startdist < enddist:
			if enddist > ModifiedFWHM:
				startdist = -startdist
			return ModifiedFWHM, startdist, centredist, MembraneLineRoi
		else: 
			if startdist > ModifiedFWHM:
				enddist = -enddist
			return ModifiedFWHM, enddist, centredist, MembraneLineRoi
	# Returns width as infinity if angle did not meet 
	# sufficent r^2 or does not contain the foci
	return float("inf"), None, None, None


def lineRotationIterator(Image, Foci, CellRoi):
	"""Gets the shortest line that can be drawn across 
	the width of a cell for a given point
	
	Takes a membrane image (ij.ImagePlus), a point (java.awt.Point) 
	and an roi defining the cell (ij.gui.Roi) and returns the width 
	of the cell at the point, the distance of the point from the 
	membrane, the distance of the point from the centre of the cell,
	and the line roi used to determine these values."""

	# Initialises a new empty results table
	RTable = ResultsTable()
	# Initialises an Analyzer object using the image 
	# and the empty results table
	An = Analyzer(Image, RTable)
	# Adds the roi to the image
	Image.setRoi(CellRoi)
	# Measures the mean of the foci
	An.measure()
	# Currently using double width as starting point 
	# for length of line
	Width = RTable.getValue("Minor", 0) * 2
	# Needs to be set to something that is always 
	# greater than all other widths found
	TempWidth = float("inf")
	# Gets the elipse angle, rounds it and gets the angle 90 degrees 
	# to it which should be the approximate width angle
	ElipseAngle = RTable.getValue("Angle", 0)
	if ElipseAngle < 0:
		ElipseAngle = 360 + ElipseAngle
	ApproxAngle = Math.round(ElipseAngle) + 90
	# Iterates through angles 10 degrees either side of the approximate angle
	for Angle in range(ApproxAngle - Angle_Range/2, ApproxAngle + (Angle_Range/2) + 1):
		FullWidHMax, membdist, centredist, MembraneLineRoi = distanceFromMembrane(
			Angle, 
			Foci, 
			Image, 
			Width
		)
		if FullWidHMax < TempWidth:
			TempWidth, TempMembDist, TempCentDist, TempWidthLine, TempAngle = (
				FullWidHMax, 
				membdist, 
				centredist, 
				MembraneLineRoi, 
				Angle
			)
	# If it did not find a workable line will return None x 4
	if TempWidth == float("inf"):
		return None, None, None, None, None
	# Otherwise will return the final values for the shortest width line
	return TempWidth, TempMembDist, TempCentDist, TempWidthLine, TempAngle

def distanceToRoiEdge(Angle, Foci, CellRoi):
	End_X, End_Y = float(Foci[0]), float(Foci[1])
	distance = 0
	while CellRoi.containsPoint(End_X, End_Y):
		distance += 1
		End_X, End_Y = getLineEndCoords(End_X, End_Y, distance, Angle)
	PoleRoi = Line(Foci[0], Foci[1], End_X, End_Y)
	PoleRoi.setStrokeWidth(1)
	return distance, PoleRoi

def getPoleDistance(Angle, CellFoci, CellRoi):
	PoleDistance_1, PoleRoi_1 = distanceToRoiEdge(
		Angle - 90, 
		CellFoci, 
		CellRoi
	)
	PoleDistance_2, PoleRoi_2 = distanceToRoiEdge(
		Angle + 90, 
		CellFoci, 
		CellRoi
	)
	
	if PoleDistance_1 < PoleDistance_2:
		return PoleDistance_1, PoleRoi_1
	else:
		return PoleDistance_2, PoleRoi_2

def getFociFocusValues(Image, Foci, Radius):
	FullWidHMax_List = []
	R_Squared_List = []
	SDevRes_List = []
	ResSumSquared_List = []
	for angle in range(0, 151, 30): # 151 as 0 == 180
		Start_X, Start_Y = getLineEndCoords(
			Foci[0], 
			Foci[1], 
			Radius, 
			angle
		)
		End_X, End_Y = getLineEndCoords(
			Foci[0], 
			Foci[1], 
			Radius, 
			angle + 180
		)
		LineRoi = Line(Start_X, Start_Y, End_X, End_Y)
		LineRoi.setStrokeWidth(3)
		Parameters, FullWidHMax, R_Squared, SDevRes = fullWidthHalfMax(
			Image, 
			LineRoi
		)
		FullWidHMax_List.append(FullWidHMax * PixelScale)
		R_Squared_List.append(R_Squared)
		SDevRes_List.append(SDevRes)
		ResSumSquared_List.append(Parameters[4])
	Mean_FWHM = sum(FullWidHMax_List)/len(FullWidHMax_List)
	DeviationList = []
	for FullWidthHalfMax in FullWidHMax_List:
		DeviationList.append((FullWidthHalfMax - Mean_FWHM)**2)
	StandardDeviation = Math.sqrt(sum(DeviationList)/len(DeviationList))
	Mean_R_Squared = sum(R_Squared_List)/len(R_Squared_List)
	Mean_SDevRes = sum(SDevRes_List)/len(SDevRes_List)
	Mean_ResSumSquared = sum(ResSumSquared_List)/len(ResSumSquared_List)
		
	return (
		Mean_FWHM, 
		StandardDeviation, 
		Mean_R_Squared, 
		Mean_SDevRes, 
		Mean_ResSumSquared
	 )

# This function will take a foci and draw lines either side of the foci.
# It will then call the FullWidthHalfMax function to get the 
# distance of the foci from the membrane
def fociMeasurements(
		CellRoiList,
		Cell_to_Points_Dict,
		SpotDict,
		ImageList,
		BackgroundList,
		Radius):
	"""Gets the distance of foci from the centre 
	of the cell and from the cell membrane
	
	Takes a list of cell shape ROIs ([ij.gui.Roi]), a dicitonary 
	containing foci in each cell using cell ROIs as keys and the 
	membrane image (ij.ImagePlus) and returns a list of dictionaries 
	with foci as keys and Cell width, distance from membrane, distance 
	from cell centre and width line roi as values """

	# Defines the dictionaries that will be returned.
	DataDict = {}
	WidthLineDict = {}
	EdgeDict = {}
	# Removes the scale from the image 
	# (needed for the coordinate system to work with plot profile)
	IJ.run(ImageList[0], "Set Scale...", "distance=0 known=0 unit=pixel")
	# Uses the list so can maintain this order going forward
	for CellNumber in range(0, len(CellRoiList)):
		try:
			CellRoi = CellRoiList[CellNumber]
			# Iterates through foci within each cell roi
			for CellFoci in Cell_to_Points_Dict[CellRoi]:
				(CellWidth, 
				MembraneDistance, 
				CentreDistance, 
				LineRoi, 
				Angle) = lineRotationIterator(
					ImageList[0], 
					CellFoci, 
					CellRoiList[CellNumber])
				
				# Checks that function was called successfully
				if not CellWidth:
					continue

				PoleDistance, PoleRoi = getPoleDistance(
					Angle, 
					CellFoci, 
					CellRoi
				)

				(Mean_FWHM, 
				StandardDeviation, 
				Mean_R_Squared, 
				Mean_SDevRes, 
				Mean_ResSumSquared) = getFociFocusValues(
					ImageList[1], 
					CellFoci, 
					Radius
				)
				
				SpotRoi = SpotDict[CellFoci]
				MeanList = []
				for Index in range(0, len(ImageList)):
					IntensityList = getRoiMeasurements(SpotRoi, ImageList[Index], ["Mean", "Max"])
					if IntensityList[1] == 65535:
						IntensityList[0] = float("inf")
					MeanList.append(IntensityList[0] - BackgroundList[Index])

				DataDict[CellFoci] = [
					CellWidth, 
					MembraneDistance, 
					CentreDistance, 
					PoleDistance, 
					MeanList[0],
					MeanList[1],
					MeanList[2],
					Mean_FWHM, 
					StandardDeviation, 
					Mean_R_Squared, 
					Mean_SDevRes, 
					Mean_ResSumSquared
				]

				WidthLineDict[CellFoci] = LineRoi
				EdgeDict[CellFoci] = PoleRoi
		# Exception called if cell has no foci 
		except(KeyError):
			continue
	return WidthLineDict, EdgeDict, DataDict

def saveRoiDictionary(
		CellRoiList, 
		Cell_to_Points_Dict, 
		RoiDictionary, 
		File_Path):
	"""Saves the roi contained within a dictionary 
	which has point objects as keys"""
	# Creates an RoiManager Instance
	Roi_Manger = RoiManager(True)
	# Uses the list so can maintain this order going forward
	for AnalysedRoi in CellRoiList:
		CellRoiName = AnalysedRoi.getName()
		# If the cell has no foci then will not save the PointRoi
		try:
			FociNumber = 0
			# Will go through the foci found in the list entry in 
			# the dictionary and gets the corresponding roi for it
			for CellFoci in Cell_to_Points_Dict[AnalysedRoi]:
				SaveRoi = RoiDictionary[CellFoci]
				if SaveRoi != None:
					SaveRoi.setName(
						CellRoiName 
						+ "-Foci_" 
						+ str(FociNumber + 1)
					)
					# Adds the point to the roi manager
					Roi_Manger.addRoi(SaveRoi)
				FociNumber += 1
		# If the cell has no foci then will not save the PointRoi
		except(KeyError):
			pass
	# Saves the roi as a zip file using the image filename 
	Roi_Manger.save(File_Path)
	Roi_Manger.close()

def saveData(
		ResultsFilename,
		OutputTable, 
		ImageName, 
		CellRoiList, 
		Cell_to_Points_Dict, 
		OutputDictionary, 
		CellPropertiesDict,
		NucleoidCellMeans, 
		QualityDict):
	# Uses the list so can maintain this order going forward
	for AnalysedRoi in CellRoiList:
		CellRoiName = AnalysedRoi.getName()
		# If the cell has no foci then will not save the PointRoi
		try:
			FociPerCell = len(Cell_to_Points_Dict[AnalysedRoi])
			# Will go through the foci found in
			# the list entry in the dictionary
			for FociNumber in range(0, FociPerCell):
				CellFoci = Cell_to_Points_Dict[AnalysedRoi][FociNumber]
				CellWidth = OutputDictionary[CellFoci][0] * 0.065
				MembraneDistance = OutputDictionary[CellFoci][1] * 0.065
				RelativeMembraneDistance = MembraneDistance/(CellWidth/2)
				CentreDistance = OutputDictionary[CellFoci][2] * 0.065
				RelativeCentreDistance = CentreDistance/(CellWidth/2)
				try:
					PoleDistance = OutputDictionary[CellFoci][3] * 0.065
				except TypeError:
					PoleDistance = OutputDictionary[CellFoci][3]
				MembraneSpotMean = OutputDictionary[CellFoci][4]
				FociSpotMean = OutputDictionary[CellFoci][5]
				NucleoidSpotMean = OutputDictionary[CellFoci][6]
				NucleoidCellMean = NucleoidCellMeans[AnalysedRoi][0]
				NucleoidCellMax = NucleoidCellMeans[AnalysedRoi][3]
				NucleoidRelativeIntensity = NucleoidSpotMean/NucleoidCellMean
				NucleoidPercentage = NucleoidSpotMean/NucleoidCellMax
				CellMean = CellPropertiesDict[AnalysedRoi][0]
				CellArea = CellPropertiesDict[AnalysedRoi][1]
				CellLength = CellPropertiesDict[AnalysedRoi][2]
				FoldIncrease = FociSpotMean/CellMean
				TrackMateQuality = QualityDict[CellFoci]
				Mean_FWHM = OutputDictionary[CellFoci][7]
				StandardDeviation = OutputDictionary[CellFoci][8]
				Mean_R_Squared = OutputDictionary[CellFoci][9]
				Mean_SDevRes = OutputDictionary[CellFoci][10]
				Mean_ResSumSquared = OutputDictionary[CellFoci][11]

				OutputTable.addValue("Image", ImageName)
				OutputTable.addValue("Cell Number", CellRoiName)
				OutputTable.addValue("Foci Number", FociNumber + 1)
				OutputTable.addValue("Spot Count", FociPerCell)
				OutputTable.addValue("Cell and Foci Number", CellRoiName + "_Foci_" + str(FociNumber+1))
				OutputTable.addValue("Cell Width", CellWidth)
				OutputTable.addValue("Membrane Distance", MembraneDistance)
				OutputTable.addValue("Relative Membrane Distance", RelativeMembraneDistance)
				OutputTable.addValue("Centre Distance", CentreDistance)
				OutputTable.addValue("Relative Centre Distance", RelativeCentreDistance)
				OutputTable.addValue("Pole Distance", PoleDistance)
				OutputTable.addValue("Foci Mean Intensity", FociSpotMean)
				OutputTable.addValue("Whole Cell Mean", CellMean)
				OutputTable.addValue("Fold increase over Cell", FoldIncrease)
				OutputTable.addValue("Foci Mean Intensity (Membrane)", MembraneSpotMean)
				OutputTable.addValue("Foci Mean Intensity (Nucleoid)", NucleoidSpotMean)
				OutputTable.addValue("Nucleoid Relative Intensity", NucleoidRelativeIntensity)
				OutputTable.addValue("Nucleoid Percentage of Maximum", NucleoidPercentage)
				OutputTable.addValue("Mean Foci Diameter", Mean_FWHM)
				OutputTable.addValue("Foci Diameter Standard Deviation", StandardDeviation)
				OutputTable.addValue("Mean Foci Gaussian R^2", Mean_R_Squared)
				OutputTable.addValue("Mean Foci Gaussian StDev of Residuals", Mean_SDevRes)
				OutputTable.addValue("Mean Foci Gaussian Residual Sum of Squares", Mean_ResSumSquared)
				OutputTable.addValue("TrackMate Quality", TrackMateQuality)
				OutputTable.addValue("Cell Area", CellArea)
				OutputTable.addValue("Cell Length", CellLength)
				
				OutputTable.save(ResultsFilename)
				Output_Table.incrementCounter()
		except(KeyError):
			pass

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

reload(sys)
sys.setdefaultencoding('utf8')

Home_Path = Home_Folder.getPath()
IlastikProject, Raw_Image_Path, FilteredImageList = checkInputs(Home_Path)
Memb_Chan -= 1
Foci_Chan -= 1
Nuc_Chan -= 1
# Converts radius from microns to pixels
ScaledRadius = (Foci_Diameter/PixelScale)/2

# This section sets the measurements that will be used
AnalyzerClass = Analyzer()
# Gets original measurements to reset later
OriginalMeasurements = AnalyzerClass.getMeasurements()
# Sets the measurements to be used
AnalyzerClass.setMeasurements(
	Measurements.MEAN 
	+ Measurements.ELLIPSE 
	+ Measurements.AREA
	+ Measurements.MIN_MAX
)

# This section initialises variable and services------------------------------------v
# that will be needed repeatedly.

# This gets a helper class that will 
# aquire instances of different services
Con = Context()
Helper = ServiceHelper(Con)

# These next statements inputs a class which lets you load
# their respective services (may act as singletons)
# Gets the service for the status bar
Status = Helper.loadService(DefaultStatusService)

# Gets the service for the log window
Logs = Helper.loadService(StderrLogService)

# Constructs the pixel classification class using the ilastik.exe file, 
# the project file, the log service, the status service, 
# the number of allowed threads and number of allowed ram
Classification = PixelClassification(
	IlastikExe, 
	IlastikProject, 
	Logs, 
	Status, 
	Threads, 
	IlRam
)

# Gets the classification type for the selected classification options
ClassificationType = Classification.PixelPredictionType.valueOf(IlastikOutputType)
# Initialises an instance of the Roi Manager that is not shown to the user
RoiMan = RoiManager(True)
#-----------------------------------------------------------------------------------^

# This section makes the directories that will be used later-------v
# Defines all the directorys to be made in the home folder
Dirs_To_Be_Made = [
	"1-Chr_Abberation_Corrected", 
	"2-Membrane_Probability", 
	"3-Membrane_Segmented",  
	"4-Cell_ROIs", 
	"5-Foci_ROIs", 
	"6-Spot_ROIs", 
	"7-Width_ROIs", 
	"8-Polar_ROIs", 
	"9-Results"
]

# Iterates through the folders to be made and checks if they 
# already exist, if not will create them
for FutureDirectory in Dirs_To_Be_Made:
	hypothetical_path = os.path.join(Home_Path, FutureDirectory)
	if os.path.exists(hypothetical_path) == False:
		os.mkdir(hypothetical_path)
#------------------------------------------------------------------^

# This section sets up the results file for later use-v
# Opens the results file
Results_Filename = os.path.join(
	Home_Path, 
	Dirs_To_Be_Made[8], 
	"Results.csv"
)

#-----------------------------------------------------^
Output_Table = ResultsTable()

for Image_Filename in FilteredImageList:
	# Gets the path to the image
	ImageFilepath = os.path.join(Raw_Image_Path, Image_Filename)
	# Gets the filename without the extension so 
	# .tiff and .roi filenames can be generated later
	Filename_No_Extension = re.split(
		r'\.nd2$', 
		Image_Filename, 
		flags=re.IGNORECASE)[0]
	# BioFormats ImporterOptions constructor
	Options = ImporterOptions()
	# Selects the files path to be imported
	Options.setId(ImageFilepath)
	# Sets BioFormats to split channels
	Options.setSplitChannels(True)
	# Imports the image as an array of ImagePlus objects
	Import = BF.openImagePlus(Options)
	
	# Creates the filepath that all chromatic 
	# abberation corrected images will be using
	ChromCorrectionBaseFilename = os.path.join(
		Home_Path, 
		Dirs_To_Be_Made[0], 
		Filename_No_Extension
	)
	# Corrects for the chomatic abberation 
	# between the red and the green channels
	Membrane_Plus, Foci_Plus, Nucleoid_Plus = chromaticAbberationCorrection(
		Import, 
		RedChromAbCorr,
		UVChromAbCorr, 
		Memb_Chan, 
		Foci_Chan,
		Nuc_Chan,
		Membrane_Channel_Colour, 
		ChromCorrectionBaseFilename
	)

	# Uses ilastik to generate a 
	# probability map using the membrane channel
	IlastikImagePlus = generateMembraneProbability(
		Membrane_Plus, 
		Classification, 
		ClassificationType, 
		Filename_No_Extension, 
		os.path.join(Home_Path, Dirs_To_Be_Made[1])
	)

	# Segments the cells based upon the 
	# probability map generated by ilastik
	BinaryFilePath = os.path.join(
		Home_Path, 
		Dirs_To_Be_Made[2], 
		Filename_No_Extension
	)
	MembraneBinary = segmentIlastikOutput(
		IlastikImagePlus, 
		Gaussian_BlurSigma, 
		ThresholdingMethod, 
		DilationPixels, 
		WatershedBool, 
		BinaryFilePath
	)
	
	# Gets the rois from the segmented image along with a combined roi
	CellRoiFilePath = os.path.join(
		Home_Path, 
		Dirs_To_Be_Made[3], 
		Filename_No_Extension + ".zip"
	)
	Cell_Roi_List, MergedCells = getCellRoi(
		MembraneBinary, 
		Cell_Sizes, 
		Cell_Circularity, 
		CellRoiFilePath
	)

	# Gets the mean background intensity of the fluorescence channels
	Foci_Background_Mean = getBackground(
		Foci_Plus, 
		MergedCells, 
		Roi_Enlargement
	)
	Membrane_Background_Mean = getBackground(
		Membrane_Plus, 
		MergedCells, 
		Roi_Enlargement
	)
	Nucleoid_Background_Mean = getBackground(
		Nucleoid_Plus,
		MergedCells,
		Roi_Enlargement
	)

	# Gets a dictionary which has the cell roi as a key with 
	# a list of all of that cells foci as the value, 
	# along with an roi containing all foci in the image
	FociDict, Quality_Dict, Spot_Dict = getFoci(
		Foci_Plus, 
		ScaledRadius, 
		TrackMate_Quality, 
		Cell_Roi_List, 
		Roi_Enlargement
	)
	
	# Saves the foci in the foci dict
	FociSavePath = os.path.join(
		Home_Path, 
		Dirs_To_Be_Made[4], 
		Filename_No_Extension + ".zip"
	)
	saveFociRoi(Cell_Roi_List, FociDict, FociSavePath)

	# Collects all the measurements of the foci
	WidthLineRoiDict, PoleLineRoiDict, Output_Dict = fociMeasurements(
													Cell_Roi_List, 
													FociDict,
													Spot_Dict,
													[Membrane_Plus, 
													Foci_Plus,
													Nucleoid_Plus],
													[Membrane_Background_Mean, 
													Foci_Background_Mean,
													Nucleoid_Background_Mean],
													ScaledRadius)


	# Saves Spot Rois
	SpotRoiSavepath = os.path.join(
		Home_Path, 
		Dirs_To_Be_Made[5], 
		Filename_No_Extension + ".zip"
	)
	saveRoiDictionary(
		Cell_Roi_List, 
		FociDict, 
		Spot_Dict, 
		SpotRoiSavepath
	)
	# Saves Width Rois
	WidthRoiSavepath = os.path.join(
		Home_Path, 
		Dirs_To_Be_Made[6], 
		Filename_No_Extension + ".zip"
	)
	saveRoiDictionary(
		Cell_Roi_List, 
		FociDict, 
		WidthLineRoiDict, 
		WidthRoiSavepath
	)
	# Saves Rois for distance from pole
	PoleRoiSavepath = os.path.join(
		Home_Path, 
		Dirs_To_Be_Made[7], 
		Filename_No_Extension + ".zip"
	)
	saveRoiDictionary(
		Cell_Roi_List, 
		FociDict, 
		PoleLineRoiDict, 
		PoleRoiSavepath
	)

	# Gets the mean values of each cell 
	# to compare to focus intensity
	FociCellMeanDictionary = getCellMeasurements(
		Cell_Roi_List, 
		Foci_Plus, 
		Foci_Background_Mean
	)
	NucleoidCellMeanDictionary = getCellMeasurements(
		Cell_Roi_List,
		Nucleoid_Plus,
		Nucleoid_Background_Mean
	)

	# Saves all outputs to the results file
	saveData(
		Results_Filename,
		Output_Table, 
		Filename_No_Extension, 
		Cell_Roi_List, 
		FociDict, 
		Output_Dict, 
		FociCellMeanDictionary,
		NucleoidCellMeanDictionary, 
		Quality_Dict
	)

AnalyzerClass.setMeasurements(OriginalMeasurements)


