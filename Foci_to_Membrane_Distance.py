"""
<Authors> George Merces | Newcastle University | george.merces@newcastle.ac.uk
Dr James Grimshaw | Newcastle University | james.grimshaw@newcastle.ac.uk
Dr Calum Jukes | Newcastle University | calum.jukes@newcastle.ac.uk </Authors>
This Fiji macro runs in Jython and calculates the distance of spots from the membrane of rod shaped bacterial cells
It requires a trained Ilastik model and a folder within the selected home folder called "Raw_Images" containg nd2 files
"""

import os, re
from java.lang import Math, Double
from java.io import File
from ij import IJ
from ij.io import FileSaver
from ij.gui import GenericDialog, Roi, Line, PointRoi, ProfilePlot
from ij.measure import Measurements, ResultsTable, CurveFitter
from ij.plugin import Scaler, RGBStackMerge, RoiEnlarger
from ij.plugin.frame import RoiManager 
from ij.plugin.filter import Analyzer
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from org.scijava import Context
from org.scijava.app import DefaultStatusService
from org.scijava.service import ServiceHelper
from org.scijava.log import StderrLogService
from net.imglib2.img import ImagePlusAdapter
from org.ilastik.ilastik4ij.executors import PixelClassification
from net.imglib2.img.display.imagej import ImageJFunctions

# Settings----------------------------------------------------------------v
# Ilastik Settings---------------------------------------------------v
# Output that is used for segmentation
IlastikOutputType = "Probabilities"
# Location of ilastik.exe file
# IlastikExe = r'C:\New_folder\ilastik-1.4.0rc8\ilastik.exe' # Home Computer
IlastikExe = r'C:\Users\Jamie\ilastik-1.4.0\ilastik.exe' # CBCB Computer
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
# Membrane width measurement settings-v
# R^2 value that gausian fits must exceed
R2_Cutoff = 0.95
# Total number of pixels you want to offset FWHM by to match membrane
# Positive values put membrane outside of FWHM, negative brings membrane in
FWHM_Modifier = 0
#-------------------------------------^

#-------------------------------------------------------------------------^

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#-----------------------------------------------------------------------FUNCTIONS---------------------------------------------------------------------------#

def getDialogSettings(Dialog):
	"""Takes ij.gui.GenericDialog and returns a list of its inputs"""

	# Gets the path to the home folder
	Home = Dialog.getNextString()
	# This value is used for find maxima (Needs to be string to input into settings)
	MaximaSetting = str(Dialog.getNextNumber())
	# This is the scaling factor for the chomatic abberation correction
	ChromaticAbberationSetting = Dialog.getNextNumber()
	# Channel number of the membrane -1 so can be used as a list index
	MembraneChannelNumber = int(Dialog.getNextNumber()) - 1
	# Channel number of the foci -1 so can be used as a list index
	FociChannelNumber = int(Dialog.getNextNumber()) - 1
	# Gets the colours of the membrane and foci channels from the generic dialog
	MembraneChannelColour = Dialog.getNextChoice()
	FociChannelColour = Dialog.getNextChoice()

	return [Home, MaximaSetting, ChromaticAbberationSetting, MembraneChannelNumber, 
	FociChannelNumber, MembraneChannelColour, FociChannelColour]

def getProcessedSettings(Dialog, IlastikExePath):
	"""Takes ij.gui.GenericDialog and returns a list with processed inputs
	
	List contains: 
	[Home folder (str), Prominence setting (str), Chromatic abberation correction (float), 
	Membrane channel number (int), Foci channel number (int), Membrane channel colour (str), Foci channel colour (str), 
	Ilastik exe file (Java.io.File), Ilastik project file (Java.io.File), Path to raw image folder (str), List of .nd2 files ([str])]"""

	# Will get the inputs from the dialog------v
	# Checks if the dialog was cancelled
	if Dialog.wasOKed() == False:
		return False
	# Gets the settings from the generic dialog
	ReturnedList = getDialogSettings(Dialog)
	#------------------------------------------^

	# Checks for and processes the ilastik.exe file input-----------------------------------------------v
	# Checks that the path to the ilastik.exe file is correct
	if os.path.exists(IlastikExePath) == False:
		IJ.error("No ilastik.exe file found in given location. Please input into macro settings")
		return False
	# Converts the path to the exe file into a Java File object and adds it to the list that is returned
	ReturnedList.append(File(IlastikExePath))
	#---------------------------------------------------------------------------------------------------^
	
	# Gets the ilastik project file from the home folder and returns it as a Java file object-v
	# Gets all the files in the home folder
	HomeFiles = os.listdir(ReturnedList[0])
	# Gets a list of files that have the .ilp extension to find Ilastik projects
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
	IlastikProjectFile = File(os.path.join(ReturnedList[0], IlastikProjectFiles[0]))
	# Adds the project file to the list that is returned
	ReturnedList.append(IlastikProjectFile)
	#-----------------------------------------------------------------------------------------^

	# Checks that the channel numbers for membrane and foci channels are not the same-v
	if ReturnedList[5] == ReturnedList[6]:
		IJ.error("Selected channel colours cannot match")
		return False
	#---------------------------------------------------------------------------------^

	# Checks for the raw image folder and returns it-------------------------------v
	# Generates the path to the folder containing the raw images
	RawImagePath = os.path.join(ReturnedList[0], "Raw_Images")
	# Checks that a home folder was chosen that has a subfolder named "Raw_Images"
	if os.path.exists(RawImagePath) == False:
		IJ.error("Please choose a home folder containing a 'Raw_Images' subfolder")
		return False
	ReturnedList.append(RawImagePath)
	#------------------------------------------------------------------------------^

	# Gets a list of images, filters for .nd2 files and returns them as a list-v
	# Gets a list of filenames of the raw images
	RawImageList = os.listdir(RawImagePath)
	# This removes any filenames that do not have a .nd2 extension
	nd2_re_Obj = re.compile(r'\.nd2$', flags=re.IGNORECASE)
	ImageList = filter(nd2_re_Obj.search, RawImageList)
	# Checks that one or more images are present in the folder
	if len(ImageList) == 0:
		IJ.error("No nd2 files were detected")
		return False
	# Adds the filtered list of image filenames to the list thats returned
	ReturnedList.append(ImageList)
	#--------------------------------------------------------------------------^
	return ReturnedList

def scaleAndCrop(Scale_This_ImagePlus, Scaling_Factor):
	"""Takes ij.ImagePlus, scaling factor as a float and returns a scaled ij.ImagePlus which has
	been cropped down to given width/height to correct for chromatic abberation between channels"""
	
	# Gets the dimensions of the image, only Width and Height are needed
	Width, Height, nChannels, nSlices, nFrames = Scale_This_ImagePlus.getDimensions()
	# Gets the scaled width and height. Need to use java rounding to keep consistent
	NewWidth = Math.round(Width * Scaling_Factor)
	NewHeight = Math.round(Height * Scaling_Factor)
	# Runs the scale command with options being: Image, Width, Height, Depth, interpolation option
	ScaledImage = Scaler.resize(Scale_This_ImagePlus, NewWidth, NewHeight, 1, "bilinear")
	# Gets the distance in pixels from the top left corner that the ROI needs for setting in the centre
	Roi_X_Offset = Math.round((NewWidth-Width)/2)
	Roi_Y_Offset = Math.round((NewHeight-Height)/2)
	# Creates an Roi that will be used to crop the image down so they are the same size. Settings are x, y, width, height
	CropRoi = Roi(Roi_X_Offset, Roi_Y_Offset, Width, Height)
	# Applies the roi to the image
	ScaledImage.setRoi(CropRoi)
	# Crops the image to size
	CroppedImage = ScaledImage.crop()
	# Closes the upscaled image to conserve memory
	ScaledImage.close()
	# Returns the cropped image to the main body of the code
	return CroppedImage

def chromaticAbberationCorrection(ArrayofImages, ChromaticAbberationSetting, MembraneChannelNumber, FociChannelNumber, MembraneChannelColour, BaseFilepath):
	"""Corrects for chromatic abberation between red and green channels and saves the corrected images
	
	Takes images from bioformats import with split channels enabled [ImagePlus], scaling factor for chromatic abberation (float), 
	Channel number for membrane (int), and foci (int), channel colour of the membrane (str), filepath to save folder with an extensionless filename (str)
	and returns the corrected membrane and foci channels as ij.ImagePlus objects"""

	# Depending on which colour is membrane and which colour is foci will send the appropriate channel to have the chromatic abberation correction
	if MembraneChannelColour == "Red":
		Membrane_Image = scaleAndCrop(ArrayofImages[MembraneChannelNumber], ChromaticAbberationSetting)
		Foci_Image = ArrayofImages[FociChannelNumber]
	else:
		Foci_Image = scaleAndCrop(ArrayofImages[FociChannelNumber], ChromaticAbberationSetting)
		Membrane_Image = ArrayofImages[MembraneChannelNumber]
	
	# Merges the channels so the file will be saved after correction for chomatic abberation
	ChromCorrTiff = RGBStackMerge.mergeChannels([Foci_Image, Membrane_Image], True)
	# Saves the corrected membrane image in the Chr_Abberation_Corrected folder
	FileSaver(Membrane_Image).saveAsTiff(BaseFilepath + "_Membrane.tiff")
	# Saves the corrected foci image in the Chr_Abberation_Corrected folder
	FileSaver(Foci_Image).saveAsTiff(BaseFilepath + "_Foci.tiff")
	# Saves the corrected merged image in the Chr_Abberation_Corrected folder
	FileSaver(ChromCorrTiff).saveAsTiff(BaseFilepath + "_Merged.tiff")
	return Membrane_Image, Foci_Image

def generateMembraneProbability(MembraneImagePlus, Classification_Obj, Classification_Type, BaseFilename, BaseFolderpath):
	"""Runs the ilastik model on the membrane channel and saves/returns the output
	
	Takes Membrane channel (ij.ImagePlus), org.ilastik.ilastik4ij.executors.PixelClassification, PixelClassification.PixelPredictionType, 
	extensionless filename (str), path to folder where probability map will be saved (str) and returns the probability map (ij.ImagePlus)"""

	# Need to convert this from ImagePlus -> ImgPlus
	Membrane_ImgPlus = ImagePlusAdapter.wrapImgPlus(MembraneImagePlus)
	# Runs the actual classify pixels command in ilastik
	IlastikImgPlus = Classification_Obj.classifyPixels(Membrane_ImgPlus, Classification_Type)
	# Generates the filename for the membrane probability image 
	ProbFilename = BaseFilename + "_Membrane-Probability.tiff"
	# Converts the file into an ImagePlus for saving
	Ilastik_ImagePlus = ImageJFunctions.wrap(IlastikImgPlus, ProbFilename)
	# Saves the probability image based upon the membrane channel
	FileSaver(Ilastik_ImagePlus).saveAsTiff(os.path.join(BaseFolderpath, ProbFilename))
	return Ilastik_ImagePlus

def segmentIlastikOutput(Ilastik_ImagePlus, Gaussian_Setting, Thesholding_Method, Dilation_Setting, Watershed_Bool, BaseFilepath):
	"""Segements the probability map generated by ilastik and saves/returns the binary image
	
	Takes probability map (ij.ImagePlus), Gaussian sigma (int/bool), Thresholding method (str), number of pixels to dilate the binary (int), 
	bool for whether to apply a watershed, filepath to save folder with an extensionless filename (str) and returns the binary image (ij.ImagePlus)"""

	# If gaussian blur is enabled will perform it with the sigma value given in macro settings
	if type(Gaussian_Setting) == int: 
		IJ.run(Ilastik_ImagePlus, "Gaussian Blur...", "sigma=" + str(Gaussian_Setting))
	# Sets the threshold based upon the thesholding method in the settings
	IJ.setAutoThreshold(Ilastik_ImagePlus, Thesholding_Method + " dark")
	# Applys the thresholding to get a binary image
	IJ.run(Ilastik_ImagePlus, "Convert to Mask", "")
	# This loop will run the dilate command as many times as the number of dilation pixels specified in settings
	IterDil = 0
	while IterDil < Dilation_Setting:
		IJ.run(Ilastik_ImagePlus, "Dilate", "")
		IterDil += 1
	# Will run the watershed processing step if enabled in the settings
	if Watershed_Bool == True:
		IJ.run(Ilastik_ImagePlus, "Watershed", "")
	# Saves the segmented membrane image
	FileSaver(Ilastik_ImagePlus).saveAsTiff(BaseFilepath + "_Semented-Membrane.tiff")
	return Ilastik_ImagePlus


def getCellRoi(Binary_Image, Cell_Size_Setting, Cell_Circularity_Setting, File_Path):
	"""Runs the analyze particles command on a binary image and saves/returns the resulting rois and a combined roi
	
	Takes Binary image (ij.ImagePlus), Min/Max size (str), Min/Max circularity (str), file path to save the rois"""

	# Runs the analyze particles command to get cell ROI. Done by adding to the overlay in order to not have ROIManger shown to user
	IJ.run(Binary_Image, "Analyze Particles...", "Size=" + Cell_Size_Setting + " circularity=" + Cell_Circularity_Setting + " clear overlay exclude")
	# Gets the Overlayed ROIs from analyze particles
	Overlayed_Rois = Binary_Image.getOverlay()
	# Takes the overlay and turns it into an array of ROI
	CellRoiList = Overlayed_Rois.toArray()
	# Removes this overlay to clean up the image
	IJ.run(Binary_Image, "Remove Overlay", "")
	Roi_Manger = RoiManager(True)
	# Adds all the ROI to the ROI Manager
	for CellNumber in range(0, len(CellRoiList)):
		# Renames the roi to give it a name to trace data back to individual cells
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

def getRoiMean(SampleRoi, Image):
	"""Gets the mean value of the provided Roi for the given image
	
	Takes an ij.gui.Roi and an ij.ImagePlus and returns a double"""
	# Initialises a new empty results table
	RTable = ResultsTable()
	# Initialises an Analyzer object using the image and the empty results table
	An = Analyzer(Image, RTable)
	Image.setRoi(SampleRoi)
	An.measure()
	MeanValue = RTable.getValue("Mean", 0)
	RTable.reset()
	return MeanValue

def deleteBackground(Image, MergedRois, PixelIncrease, File_Path):
	"""Deletes the pixels no found within a certain distance of segmented cells
	
	Takes image to be subtracted (ij.ImagePlus), Merged cell rois (ij.gui.Roi), 
	Number of pixels to enlarge rois by (int), file path where subtracted image will be saved (str)"""

	# Enlarges the rois before inversion so foci just outside the membrane are not recognised
	enlarged_rois = RoiEnlarger().enlarge(MergedRois, PixelIncrease)
	# Assigns the ROI to the foci image
	Image.setRoi(enlarged_rois)
	# Inverts the cell roi so selecting the background
	IJ.run(Image, "Make Inverse", "")
	InvertedRoi = Image.getRoi()
	# Gets the mean value of the background before it is set to zero
	BackgroundMean = getRoiMean(InvertedRoi, Image)
	# Sets the background colour (which will apply when deleted) to 0
	IJ.setBackgroundColor(0, 0, 0)
	# Deletes the area outside of the cells
	IJ.run(Image, "Clear", "slice")
	# Saves the background subtracted image
	FileSaver(Image).saveAsTiff(File_Path)
	# Clears the roi from the image
	Image.resetRoi()
	return Image, BackgroundMean

def rasteriseOutline(x_array, y_array):
	"""Takes pair of integer lists corresponding to x/y coordinates of a polygon and returns 
	pair of integer lists corresponding to coordinates for the whole outline of the polygon"""

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
			slope = float(y_array[coordIndex] - y_array[coordIndex - 1])/float(x_array[coordIndex] - (x_array[coordIndex - 1]))
			# Calculates the intercept of this slope which is needed for calculating the final value
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
		# If two x coordinates are equal then will get a zero division error so this calculation is instead used
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
				# Appends both values to their respective lists (x value will stay the same)
				connected_x.append(x_array[coordIndex])
				connected_y.append(Incrementing_Y)
		return connected_x, connected_y

def distanceBetweenPoints(X1, Y1, X2, Y2): #Check whether need to use -ve Y values for this
	xdiff = X1 - X2
	ydiff = Y1 - Y2
	Distance = Math.sqrt((xdiff*xdiff) + (ydiff*ydiff))
	return Distance


def nearestNeighbourPoints(PolyRoi, MarkerPoint):
	"""Takes ij.gui.PolygonRoi and ij.gui.PointRoi and returns the closest distance between the PointRoi and the outer edge of the PolygonRoi as a float"""

	# Sets the initial distance to infinity so will always be lower than one value
	distance = float('inf')
	# Gets the outline as a polygon
	Poly = PolyRoi.getPolygon()
	# Calls the rasteriseOutline function to connect points in polygon
	x_arr, y_arr = rasteriseOutline(Poly.xpoints, Poly.ypoints)
	# This loop determines the distance of each coordinate in the polygon from the given point
	for arrayindex in range(0, len(x_arr)):
		tempdist = distanceBetweenPoints(MarkerPoint.x, MarkerPoint.y, x_arr[arrayindex], y_arr[arrayindex])
		if tempdist < distance:
			distance = tempdist
	return distance

def checkSoleContainer(Point, enlarged_CellRoi, CellRoi, Cell_to_Points_Dict, Points_to_Cell_Dict):
	"""Checks if the given foci is solely contained by/closest to the given cell roi
	
	Takes foci (java.awt.Point), enlarged cell roi (ij.gui.Roi), cell roi (ij.gui.Roi), Cell_to_Points_Dict (dict), Points_to_Cell_Dict (dict)"""

	# Checks if the point falls within the enlarged roi
	if enlarged_CellRoi.contains(Point.x, Point.y):
		# Checks if the point has already been assigned to another cell
		if Point not in Points_to_Cell_Dict:
			return True, Cell_to_Points_Dict
		# Gets the other cells roi from the dictionary with the point as the key
		OtherCellRoi = Points_to_Cell_Dict[Point]
		# Checks if the point is within the un-enlarged (current) roi
		if CellRoi.contains(Point.x, Point.y):
			# Removes the point from being assigned to the old cell roi
			Cell_to_Points_Dict[OtherCellRoi].remove(Point)
			return True, Cell_to_Points_Dict
		# Checks if instead the point is within the un-enlarged (old) roi
		elif OtherCellRoi.contains(Point.x, Point.y):
			# If true then no dictionaries need altering
			return False, Cell_to_Points_Dict
		# If it falls within neither roi then will determine the closest roi to it
		# Calls the nearestNeighbourPoints function to get the closest distance to one of the points
		CellDistance = nearestNeighbourPoints(CellRoi, Point)
		OtherCellDistance = nearestNeighbourPoints(OtherCellRoi, Point)
		# Checks if the new roi is closest
		if CellDistance < OtherCellDistance:
			# Removes the point from being assigned to the old cell roi
			Cell_to_Points_Dict[OtherCellRoi].remove(Point)
			return True, Cell_to_Points_Dict
	else:
		return False, Cell_to_Points_Dict

def getFoci(Image, MaximaSetting, CellRoiList, PixelIncrease):
	"""Gets the foci found within each cell roi
	
	Takes the foci image (ij.ImagePlus), Prominence setting (str), array of cell rois ([ij.gui.Roi]), Number of pixels to enlarge rois by (int) 
	and returns a dictionary with cell rois as keys (ij.gui.Roi) and list of points as values ([java.awt.Point]), an roi containing all foci (ij.gui.PointRoi)"""

	# Finds the foci using find maxima command and chosen prominence
	IJ.run(Image, "Find Maxima...", "prominence=" + MaximaSetting + " output=[Point Selection]")
	# Gets the points roi from the image
	AllMaxima = Image.getRoi()
	# Gets the points within this roi as an array
	PointArray = AllMaxima.getContainedPoints()
	# This section assigns the foci to individual cells-------------------------------------------------------------v
	# Creates dictionaries to get per cell points
	Cell_to_Points_Dict = {}
	Points_to_Cell_Dict = {}
	# This loop will go through each roi and determine which points are within an enlarged form of it
	# If the point falls into the expanded area of multiple foci. It will be associated with the closest cell 
	for CellRoi in CellRoiList:
		# Enlarges the roi so foci just outside the membrane are recognised
		enlarged_CellRoi = RoiEnlarger().enlarge(CellRoi, PixelIncrease)
		# Iterates through all of the points for every cell
		for Point in PointArray:
			# Checks whether the point should be assigned to this cell
			SaveBool, Cell_to_Points_Dict = checkSoleContainer(Point, enlarged_CellRoi, CellRoi, Cell_to_Points_Dict, Points_to_Cell_Dict)
			if SaveBool:
				# Adds the roi to the dictionary using points as keys
				Points_to_Cell_Dict[Point] = CellRoi
				# Try except will speed up this process
				try:
					# Appends this point to the dictionary using the roi as a key
					Cell_to_Points_Dict[CellRoi].append(Point)
				except(KeyError):
					# Adds a list containing this point to the dictionary using the roi as a key
					Cell_to_Points_Dict[CellRoi] = [Point]
	return Cell_to_Points_Dict, AllMaxima

def saveFociRoi(CellRoiList, Cell_to_Points_Dict, AllMaxima, File_Path):
	"""Saves the foci in the provided dictionary as roi
	
	Takes array of cell rois ([ij.gui.Roi]), a dictionary with cell rois as keys and list of their containing foci as points the values ({ij.gui.Roi = [java.awt.Point]}), 
	an roi containing all foci (ij.gui.PointRoi), file path to save the roi (str)"""

	# Creates an RoiManager Instance
	Roi_Manger = RoiManager(True)
	# Sets the name of the roi containing all foci
	AllMaxima.setName("AllPoints")
	# Adds the roi to the roi manager
	Roi_Manger.addRoi(AllMaxima)
	# Uses the list so can maintain this order going forward
	for AnalysedRoi in CellRoiList:
		# Sets the name of the roi to match the cell roi name
		CellRoiName = AnalysedRoi.getName()
		# If the cell has no foci then will not save the PointRoi
		try:
			FociNumber = 0
			# Will go through the foci found in the list entry in the dictionary and add them to a single PointRoi
			for CellFoci in Cell_to_Points_Dict[AnalysedRoi]:
				# Creates an empty PointRoi obj
				SaveRoi = PointRoi(CellFoci.x, CellFoci.y)
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

def getCellMean(RoiList, Image, Background):
	OutputDict = {}
	for CellRoi in RoiList:
		OutputDict[CellRoi] = getRoiMean(CellRoi, Image) - Background
	return OutputDict

def getLineEndCoords(X, Y, Length, Angle):
	"""Takes the x and y the starting point of the line, the desired length and angle of a line and returns the end coordinates of that line"""
	# Need to convert the angle to radians 
	Radian = Math.toRadians(Angle)
	# These equations generate the end coordinates and returns them
	return X + (Length * Math.cos(Radian)), abs(-Y + (Length * Math.sin(Radian)))


def fullWidthHalfMax(Image, LineRoi):
	"""Gets the fluorescence intesnsity profile of a line, fits a gaussian curve and calculates the full width half maximum of that curve
	
	Takes an Image (ij.ImagePlus) and a Line roi (ij.gui.Line) and returns the parameters: 
	minimum, maximum, centre of the peak, standard deviation, R^2 ([double]) 
	and the full width half maximum of the gaussian curve (double)"""
	
	# Adds the line roi to the image
	Image.setRoi(LineRoi)
	# Generates the fluorescence intesnsity profile
	Fluor_Profile = ProfilePlot(Image)
	# Gets the Plot obj 
	Fluor_Plot = Fluor_Profile.getPlot()
	# Pulls the values for X from the plot (as float array)
	X_Values = Fluor_Plot.getXValues()
	# Pulls the values for Y straight from the ProfilePlot (as a double array)
	Y_Values = Fluor_Profile.getProfile()
	# Converts the float array of X values into a double array (needed for the curve fitting)
	X_Double = []
	for Value in X_Values:
		X_Double.append(Double(Value))
	# Fits the gaussian curve
	Fit = CurveFitter(X_Double, Y_Values)
	Fit.doFit(Fit.GAUSSIAN)
	# Gets the parameters from the fit as a double array
	# Parameters are minimum, maximum, centre of the peak, standard deviation, R^2
	Parameters = Fit.getParams()
	# Calculates the full width half max from the standard deviation parameter
	FullWidHMax = Parameters[3] * (2 * Math.sqrt(2 * Math.log(2)))
	# Clears the roi from the image
	Image.resetRoi()
	return Parameters, FullWidHMax

def distanceFromMembrane(Angle, Foci, Image, Length):
	"""Gets the width of the membrane and distance of foci from membrane edges and centre of the cell"""
	# Need to make sure when getting the other side of the line the angle isnt going over 360 degrees 
	if Angle > 180:
		AngleMod = -180
	else:
		AngleMod = 180
	# Gets one side of the extended line 
	Start_X, Start_Y = getLineEndCoords(Foci.x, Foci.y, Length, Angle)
	# Gets the other side of the extended line
	End_X, End_Y = getLineEndCoords(Foci.x, Foci.y, Length, Angle + AngleMod)
	# Generates the line roi
	TempLineRoi = Line(Start_X, Start_Y, End_X, End_Y)
	# Sets the width to 3 pixels wide so there is some averaging on the fluorescence intensity profile
	TempLineRoi.setWidth(3)
	# Gets the parameters of a gaussian fit (minimum, maximum, centre of the peak, standard deviation, R^2) and the full width half maximum
	Parameters, FullWidHMax = fullWidthHalfMax(Image, TempLineRoi)
	ModifiedFWHM = FullWidHMax + FWHM_Modifier
	# Finds the distance from the start of the line to the start of the membrane
	DistanceToMembraneStart = Parameters[2] - (ModifiedFWHM/2)
	# Gets the coordinates of the start of the membrane
	Membrane_Start_X, Membrane_Start_Y = getLineEndCoords(Start_X, Start_Y, DistanceToMembraneStart, Angle + AngleMod)
	# Gets the coordinates of the centre peak of the curve
	Membrane_Centre_X, Membrane_Centre_Y = getLineEndCoords(Membrane_Start_X, Membrane_Start_Y, ModifiedFWHM/2, Angle + AngleMod)
	# Gets the coordinates of the end of the membrane
	Membrane_End_X, Membrane_End_Y = getLineEndCoords(Membrane_Start_X, Membrane_Start_Y, ModifiedFWHM, Angle + AngleMod)
	# Generates the line roi for this  membrane  spanning line
	MembraneLineRoi = Line(Membrane_Start_X, Membrane_Start_Y, Membrane_End_X, Membrane_End_Y)
	# Sets the width of this line to 2 pixels so can check if it contains the foci properly
	MembraneLineRoi.setWidth(2)
	# Distance of the foci from the start of the membrane
	startdist = distanceBetweenPoints(Membrane_Start_X, Membrane_Start_Y, Foci.x, Foci.y)
	# Distance of the foci from the centre of the cell
	centredist = distanceBetweenPoints(Membrane_Centre_X, Membrane_Centre_Y, Foci.x, Foci.y)
	# Distance of the foci from the end of the membrane 
	enddist = distanceBetweenPoints(Membrane_End_X, Membrane_End_Y, Foci.x, Foci.y)
	# Checks whether the line has the shortest distance found so far, if the R^2 of the fit is greater than a certain thresholding and that it contains the foci
	if Parameters[4] > R2_Cutoff and MembraneLineRoi.contains(Foci.x, Foci.y):
		# Finds which distance is shorter the distance from the 
		if startdist < enddist:
			if enddist > ModifiedFWHM:
				startdist = -startdist
			return ModifiedFWHM, startdist, centredist, MembraneLineRoi
		else: 
			if startdist > ModifiedFWHM:
				enddist = -enddist
			return ModifiedFWHM, enddist, centredist, MembraneLineRoi
	# Returns width as infinity if angle did not meet sufficent r^2 or does not contain the foci
	return float("inf"), None, None, None


def lineRotationIterator(Image, Foci, CellRoi):
	"""Gets the shortest line that can be drawn across the width of a cell for a given point
	
	Takes a membrane image (ij.ImagePlus), a point (java.awt.Point) and an roi defining the cell (ij.gui.Roi) and returns the width of the cell at the point,
	the distance of the point from the membrane, the distance of the point from the centre of the cell, and the line roi used to determine these values."""

	# Initialises a new empty results table
	RTable = ResultsTable()
	# Initialises an Analyzer object using the image and the empty results table
	An = Analyzer(Image, RTable)
	# Adds the roi to the image
	Image.setRoi(CellRoi)
	# Measures the mean of the foci
	An.measure()
	# Currently using double width as starting point for length of line
	Width = RTable.getValue("Minor", 0) * 2
	# Needs to be set to something that is always greater than all other widths found
	TempWidth = float("inf")
	# Gets the elipse angle, rounds it and gets the angle 90 degrees to it which should be the approximate width angle
	ElipseAngle = RTable.getValue("Angle", 0)
	if ElipseAngle < 0:
		ElipseAngle = 360 + ElipseAngle
	ApproxAngle = Math.round(ElipseAngle) + 90
	# Iterates through angles 10 degrees either side of the approximate angle
	for Angle in range(ApproxAngle - 10, ApproxAngle + 11):
		FullWidHMax, membdist, centredist, MembraneLineRoi = distanceFromMembrane(Angle, Foci, Image, Width)
		if FullWidHMax < TempWidth:
			TempWidth, TempMembDist, TempCentDist, TempWidthLine, TempAngle = FullWidHMax, membdist, centredist, MembraneLineRoi, Angle
	# If it did not find a workable line will return None x 4
	if TempWidth == float("inf"):
		return None, None, None, None, None
	# Otherwise will return the final values for the shortest width line
	return TempWidth, TempMembDist, TempCentDist, TempWidthLine, TempAngle

def getFociFocusValue(Image, Foci):
	FociRoi = PointRoi(Foci.x, Foci.y)
	FociMean = getRoiMean(FociRoi, Image)
	# Enlarges the point by one pixel to form a 3x3 roi
	EnlargedRoi = RoiEnlarger().enlarge(FociRoi, 1)
	EnlargedMean = getRoiMean(EnlargedRoi, Image)
	# Divides the expanded roi mean with the mean at the centre of the foci
	RelativeMean = EnlargedMean/FociMean
	# An.setMeasurements(OriginalMeasurements)
	return FociMean, EnlargedMean

def poleDistance(Angle, Foci, CellRoi):
	End_X, End_Y = float(Foci.x), float(Foci.y)
	distance = 0
	while CellRoi.containsPoint(End_X, End_Y):
		distance += 1
		End_X, End_Y = getLineEndCoords(End_X, End_Y, distance, Angle)
	PoleRoi = Line(Foci.x, Foci.y, End_X, End_Y)
	PoleRoi.setWidth(1)
	return distance, PoleRoi

# This function will take a foci and draw lines either side of the foci.
# It will then call the FullWidthHalfMax function to get the distance of the foci from the membrane
def fociMeasurements(CellRoiList, Cell_to_Points_Dict, MembraneImage, FociImage, BackgroundMean):
	"""Gets the distance of foci from the centre of the cell and from the cell membrane
	
	Takes a list of cell shape ROIs ([ij.gui.Roi]), a dicitonary containing foci in each cell using cell ROIs as keys and the membrane image (ij.ImagePlus)
	and returns a list of dictionaries with foci as keys and Cell width, distance from membrane, distance from cell centre and width line roi as values """

	# Defines the dictionaries that will be returned.
	DataDict = {}
	# WidthDict = {}
	# MembraneDistanceDict = {}
	# CentreDistanceDict = {}
	WidthLineDict = {}
	# PoleDistanceDict = {}
	EdgeDict = {}
	# RelativeMeanDict = {}
	# Removes the scale from the image (needed for the coordinate system to work with plot profile)
	IJ.run(MembraneImage, "Set Scale...", "distance=0 known=0 unit=pixel")
	# Uses the list so can maintain this order going forward
	for CellNumber in range(0, len(CellRoiList)):
		try:
			CellRoi = CellRoiList[CellNumber]
			# Iterates through foci within each cell roi
			for CellFoci in Cell_to_Points_Dict[CellRoi]:
				CellWidth, MembraneDistance, CentreDistance, LineRoi, Angle = lineRotationIterator(MembraneImage, CellFoci, CellRoiList[CellNumber])
				# Checks that function was called successfully
				if CellWidth:
					PoleDistance_1, PoleRoi_1 = poleDistance(Angle - 90, CellFoci, CellRoi)
					PoleDistance_2, PoleRoi_2 = poleDistance(Angle + 90, CellFoci, CellRoi)
					if PoleDistance_1 < PoleDistance_2:
						PoleDistance = PoleDistance_1
						PoleRoi = PoleRoi_1
					else:
						PoleDistance = PoleDistance_2
						PoleRoi = PoleRoi_2

					FociMean, EnlargedMean = getFociFocusValue(FociImage, CellFoci)

					DataDict[CellFoci] = [CellWidth, MembraneDistance, CentreDistance, 
					PoleDistance, FociMean - BackgroundMean, EnlargedMean - BackgroundMean]
					WidthLineDict[CellFoci] = LineRoi
					EdgeDict[CellFoci] = PoleRoi

		# Exception called if cell has no foci 
		except(KeyError):
			pass
	return WidthLineDict, EdgeDict, DataDict

def saveRoiDictionary(CellRoiList, Cell_to_Points_Dict, RoiDictionary, File_Path):
	"""Saves the roi contained within a dictionary which has point objects as keys"""
	# Creates an RoiManager Instance
	Roi_Manger = RoiManager(True)
	# Uses the list so can maintain this order going forward
	for AnalysedRoi in CellRoiList:
		CellRoiName = AnalysedRoi.getName()
		# If the cell has no foci then will not save the PointRoi
		try:
			FociNumber = 0
			# Will go through the foci found in the list entry in the dictionary and gets the corresponding roi for it
			for CellFoci in Cell_to_Points_Dict[AnalysedRoi]:
				SaveRoi = RoiDictionary[CellFoci]
				if SaveRoi != None:
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

def saveData(ResultsFile, ImageName, CellRoiList, Cell_to_Points_Dict, OutputDictionary, MeanDict):
	# Uses the list so can maintain this order going forward
	for AnalysedRoi in CellRoiList:
		CellRoiName = AnalysedRoi.getName()
		# If the cell has no foci then will not save the PointRoi
		try:
			FociPerCell = len(Cell_to_Points_Dict[AnalysedRoi])
			# Will go through the foci found in the list entry in the dictionary
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
				FociMean = OutputDictionary[CellFoci][4]
				EnlargedMean = OutputDictionary[CellFoci][5]
				RelativeMean = EnlargedMean/FociMean
				CellMean = MeanDict[AnalysedRoi]
				FoldIncrease = FociMean/CellMean
				
				LineList = [ImageName, CellRoiName, str(FociNumber+1), str(FociPerCell), str(CellWidth), str(MembraneDistance), 
				str(RelativeMembraneDistance), str(CentreDistance), str(RelativeCentreDistance), str(PoleDistance), 
				str(FociMean), str(EnlargedMean), str(RelativeMean), str(CellMean), str(FoldIncrease), '\n']

				ResultsFile.write(",".join(LineList))
		except(KeyError):
			pass

#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

# This section defines the dialog window and shows it to the user-------------------------------------------------------------------v
gd = GenericDialog("Foci to Membrane Distance")
gd.addDirectoryField("Home_Folder:", r'R:\Calum_Jukes\New Code Test') # CBCB Computer
# gd.addDirectoryField("Home_Folder:", r'V:\Documents\Education and Work\Calum\New Macro Test') # Home Computer
gd.addNumericField("Prominence:", 500.00)
gd.addNumericField("Chromatic_Abberation_Scaling_Factor:", 1.0018)
gd.addNumericField("Membrane_Channel:", 3)
Colour_Array = ["UV", "Green", "Red"]
gd.addChoice("Membrane_Channel_Colour:", Colour_Array, "Red")
gd.addNumericField("Foci_Channel:", 2)
gd.addChoice("Foci_Channel_Colour:", Colour_Array, "Green")
# gd.addNumericField("Nucleoid_Channel:", 4, 1, 3) # Commented out untill know what doing with nuclear channel
# gd.addChoice("Nucleoid_Channel_Colour:", Colour_Array, "UV") # Commented out untill know what doing with nuclear channel
gd.showDialog()
#-----------------------------------------------------------------------------------------------------------------------------------^

# Gets the settings from the dialog (which undergo some initial processing)
DialogSettings = getProcessedSettings(gd, IlastikExe)

# DialogSettings will be equal to False if certain conditions are not met so will not proceed
if DialogSettings:
	# Gets the processed settings from the list outputted by getProcessedSettings function
	(Home_Path, Prominence, ChomAbCorr, 
  	Memb_Chan, Foci_Chan, 
	Membrane_Channel_Colour, Foci_Channel_Colour, IlastikExeFile, 
	IlastikProject, Raw_Image_Path, FilteredImageList) = DialogSettings
	
	AnalyzerClass = Analyzer()
	OriginalMeasurements = AnalyzerClass.getMeasurements()
	AnalyzerClass.setMeasurements(Measurements.MEAN + Measurements.ELLIPSE)

	# This section initialises variable and services that will be needed repeatedly-----------------------------v
	# This gets a helper class that will aquire instances of different services
	Con = Context()
	Helper = ServiceHelper(Con)
	# These next statements inputs a class which lets you load their respective services (may act as singletons)
	# Gets the service for the status bar
	Status = Helper.loadService(DefaultStatusService)
	# Gets the service for the log window
	Logs = Helper.loadService(StderrLogService)
	# Constructs the pixel classification class using the ilastik.exe file, the project file, the log service, 
	# the status service, the number of allowed threads and number of allowed ram
	Classification = PixelClassification(IlastikExeFile, IlastikProject, Logs, Status, Threads, IlRam)
	# Gets the classification type for the selected classification options
	ClassificationType = Classification.PixelPredictionType.valueOf(IlastikOutputType)
	# Initialises an instance of the Roi Manager that is not shown to the user
	RoiMan = RoiManager(True)
	#-----------------------------------------------------------------------------------------------------------^

	# This section makes the directories that will be used later-----------------------------------------------------------------------------v
	# Defines all the directorys to be made in the home folder
	Dirs_To_Be_Made = ["1-Chr_Abberation_Corrected", "2-Membrane_Probability", "3-Membrane_Segmented", 
	"4-Foci_Subtracted", "5-Cell_ROIs", "6-Foci_ROIs", "7-Width_ROIs", "8-Polar_ROIs", "9-Results"]
	# Iterates through the folders to be made and checks if they already exist, if not will create them
	for FutureDirectory in Dirs_To_Be_Made:
		hypothetical_path = os.path.join(Home_Path, FutureDirectory)
		if os.path.exists(hypothetical_path) == False:
			os.mkdir(hypothetical_path)
	#----------------------------------------------------------------------------------------------------------------------------------------^
	# Opens the results file
	ResultsCSV = open(os.path.join(Home_Path, Dirs_To_Be_Made[8], "Results.csv"), "w")

	ColumnList = ["Image", "Cell Number", "Foci Number", "Spot Count", "Cell Width", "Membrane Distance", 
	"Relative Membrane Distance", "Centre Distance", "Relative Centre Distance", "Pole Distance", 
	"Centre Pixel Intensity", "Expanded Focus Mean", "Relative Mean", "Whole Cell Mean", "Fold increase over Cell" "\n"]

	# Writes the column headings to the top of the file
	ResultsCSV.write(",".join(ColumnList))

	for Image_Filename in FilteredImageList:
		# Gets the path to the image
		ImageFilepath = os.path.join(Raw_Image_Path, Image_Filename)
		# Gets the filename without the extension so .tiff and .roi filenames can be generated later
		Filename_No_Extension = re.split(r'\.nd2$', Image_Filename, flags=re.IGNORECASE)[0]
		# BioFormats ImporterOptions constructor
		Options = ImporterOptions()
		# Selects the files path to be imported
		Options.setId(ImageFilepath)
		# Sets BioFormats to split channels
		Options.setSplitChannels(True)
		# Imports the image as an array of ImagePlus objects
		Import = BF.openImagePlus(Options)
		
		# Creates the filepath that all chromatic abberation corrected images will be using
		ChromCorrectionBaseFilename = os.path.join(Home_Path, Dirs_To_Be_Made[0], Filename_No_Extension)	
		# Corrects for the chomatic abberation between the red and the green channels
		Membrane_Plus, Foci_Plus = chromaticAbberationCorrection(Import, ChomAbCorr, Memb_Chan, Foci_Chan, Membrane_Channel_Colour, ChromCorrectionBaseFilename)

		# Uses ilastik to generate a probability map using the membrane channel
		IlastikImagePlus = generateMembraneProbability(Membrane_Plus, Classification, ClassificationType, Filename_No_Extension, os.path.join(Home_Path, Dirs_To_Be_Made[1]))

		# Segments the cells based upon the probability map generated by ilastik
		MembraneBinary = segmentIlastikOutput(IlastikImagePlus, Gaussian_BlurSigma, ThresholdingMethod, DilationPixels, WatershedBool, 
		       									os.path.join(Home_Path, Dirs_To_Be_Made[2], Filename_No_Extension))
		
		# Gets the rois from the segmented image along with a combined roi
		Cell_Roi_List, MergedCells = getCellRoi(MembraneBinary, Cell_Sizes, Cell_Circularity, os.path.join(Home_Path, Dirs_To_Be_Made[4], Filename_No_Extension + ".zip"))

		# Sets any pixels outside of cells to 0
		Subbed_Foci_Plus, Background_Mean = deleteBackground(Foci_Plus, MergedCells, Roi_Enlargement, os.path.join(Home_Path, Dirs_To_Be_Made[3], Filename_No_Extension + "_Foci_Background_Sub.tiff"))

		# Gets a dictionary which has the cell roi as a key with a list of all of that cells foci as the value, along with an roi containing all foci in the image
		FociDict, MaximaRoi = getFoci(Subbed_Foci_Plus, Prominence, Cell_Roi_List, Roi_Enlargement)
		
		# Saves the foci in the foci dict
		FociSavePath = os.path.join(Home_Path, Dirs_To_Be_Made[5], Filename_No_Extension + ".zip")
		saveFociRoi(Cell_Roi_List, FociDict, MaximaRoi, FociSavePath)

		# Collects all the measurements of the foci
		WidthLineRoiDict, PoleLineRoiDict, Output_Dict = fociMeasurements(Cell_Roi_List, FociDict, Membrane_Plus, Foci_Plus, Background_Mean)

		# Saves width Rois and rois are the edges of the poles
		WidthRoiSavepath = os.path.join(Home_Path, Dirs_To_Be_Made[6], Filename_No_Extension + ".zip")
		saveRoiDictionary(Cell_Roi_List, FociDict, WidthLineRoiDict, WidthRoiSavepath)
		PoleRoiSavepath = os.path.join(Home_Path, Dirs_To_Be_Made[7], Filename_No_Extension + ".zip")
		saveRoiDictionary(Cell_Roi_List, FociDict, PoleLineRoiDict, PoleRoiSavepath)

		# Gets the mean values of each cell to compare to focus intensity
		CellMeanDictionary = getCellMean(Cell_Roi_List, Subbed_Foci_Plus, Background_Mean)
		saveData(ResultsCSV, Filename_No_Extension, Cell_Roi_List, FociDict, Output_Dict, CellMeanDictionary)
	
	AnalyzerClass.setMeasurements(OriginalMeasurements)
	ResultsCSV.close()


