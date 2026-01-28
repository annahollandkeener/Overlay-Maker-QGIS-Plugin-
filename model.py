#IMPORTS
import os
from qgis.core import QgsVectorLayer, QgsCoordinateReferenceSystem, QgsField, QgsVectorFileWriter, QgsCoordinateTransformContext, QgsMessageLog, Qgis
from qgis.PyQt.QtCore import QVariant
import processing
from processing.core.Processing import Processing
from qgis.core import QgsRasterLayer, QgsProject
from qgis.analysis import QgsRasterCalculatorEntry, QgsRasterCalculator
from osgeo import gdal
import pandas as pd
import matplotlib.pyplot as plt
from qgis.core import edit, QgsVectorDataProvider, QgsVariantUtils
import math
from PyQt5.QtCore import QMetaType 

from . import testFunctions



#MANUAL INPUTS
dem = ''
blocks = ''
overlay = ''
outputFolder = ''

class Model():

    def __init__(self, app):
        self.app = app
        

    def existingFolderHandler(self, outputFolder, subFolder = str):
        QgsMessageLog.logMessage(f"WERE IN THE EXISTING FOLDER FUNCTION", "Overlay Maker", Qgis.Info)

        newFolder = outputFolder + "/" + subFolder


        if not os.path.exists(newFolder):
            QgsMessageLog.logMessage(f"OUTPUTFOLDER: {outputFolder}, subFOLDER: {subFolder}", "Overlay Maker", Qgis.Info)

            os.makedirs(newFolder)
            QgsMessageLog.logMessage(f"WERE IN THE EXISTING FOLDER FUNCTION", "Overlay Maker", Qgis.Info)

            return newFolder
        else:
            # Folder already exists, find a unique name by appending a number
            QgsMessageLog.logMessage(f"FOLDER EXISTS...OUTPUTFOLDER: {outputFolder}, subFOLDER: {subFolder}", "Overlay Maker", Qgis.Info)

            folder_name, _ = os.path.splitext(newFolder) # Split to handle potential extensions, though not typical for folders
            
            counter = 1

            folderExists = True

            while folderExists == True:

                new_folder_path = f"{folder_name} ({counter})"

                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)

                    outputFolder = new_folder_path

                    folderExists = False

                    return new_folder_path

                else:
                    counter += 1

        


    #RASTER SUBTRACTOR: Creates a raster overlay
    def rasterSubtractor(self, dem, waterTable, outputFolder, opt = 0):
        QgsMessageLog.logMessage("\n~ Performing Raster Subtraction ~", "Overlay Maker", Qgis.Info)


        print("\n~ Performing Raster Subtraction ~")
        #subtracting flat raster of blocks from DEM using raster calculator
        if outputFolder == 'TEMPORARY_OUTPUT':
            outputPath = 'TEMPORARY_OUTPUT'
        else:
            QgsMessageLog.logMessage(f"WERE IN THE ELSE", "Overlay Maker", Qgis.Info)
            QgsMessageLog.logMessage(f"\nMake folder inputs = {outputFolder}", "Overlay Maker", Qgis.Info)


            outputFolderPath = self.existingFolderHandler(outputFolder, "Overlay")
            
            QgsMessageLog.logMessage(f"WERE AFTER THE FOLDER COMMAND", "Overlay Maker", Qgis.Info)

            outputPath = outputFolderPath 
            

        alignedDEMOutput = outputPath + "/alignedDEM"

        QgsMessageLog.logMessage(f"alignedDEMOutput: {alignedDEMOutput}", "Overlay Maker", Qgis.Info)

        alignedDEMCalc = processing.run("gdal:cliprasterbyextent", {'INPUT':dem,'PROJWIN':waterTable,'OVERCRS':False,'NODATA':None,'OPTIONS':None,'DATA_TYPE':0,'EXTRA':'','OUTPUT':alignedDEMOutput + ".tif"})

        params = {
            'INPUT_A': alignedDEMCalc['OUTPUT'], 'BAND_A': 1,
            'INPUT_B': waterTable, 'BAND_B': 1,
            'FORMULA': 'A - B',
            'OUTPUT': outputPath + "/Overlay" + str(opt) + ".tif",
            'CELLSIZE': 3,
            'EXTENT': QgsRasterLayer(waterTable, "waterTable").extent()
        }

        result = processing.run("gdal:rastercalculator", params)

        QgsMessageLog.logMessage(f"OVERLAY GENERATED: '{outputPath}'", "Overlay Maker", Qgis.Info)

        if outputFolder == 'TEMPORARY_OUTPUT':
            return result['OUTPUT']
        else:
            newOutputPath =  outputPath + "/Overlay" + str(opt) + ".tif"
            return newOutputPath

    #RASTER HISTOGRAM GENERATOR: Creates a histogram for an overlay/raster for specified blocks
    def rasterHist(self, overlay, blocks, progressFolder, outputFolder, reclass = None, histPlotName = str):
        QgsMessageLog.logMessage("\n~ Performing Raster Histogram Generation ~", "Overlay Maker", Qgis.Info)

        if progressFolder == None:
            progressFolderPath = self.existingFolderHandler(outputFolder, "Histogram")
            progressFolder = progressFolderPath
        
        #if reclass == None:
            #reclass = ['-1000','0','1','0','1','2','1','2','3','2','3','4','3','1000','5']
        
        reclass = ['-1000','-1','1','-1','-.5','2','-.5','0','3','0','1','4','1','2','5','2','3','6','3','4','7','4','1000','8']
        
        ranges = ['< -1', '-1 to -.5', '-.5 to 0', '0 to 1', '1 to 2', '2 to 3', '3 to 4', '> 4']
            
        baseName = os.path.basename(overlay).rsplit(".", 1)[0] + "_"

        #reclassifying overlay based on block zones
        reclassRast = processing.run("native:reclassifybytable", {'INPUT_RASTER':overlay,'RASTER_BAND':1,'TABLE':reclass,'NO_DATA':-9999,'RANGE_BOUNDARIES':0,'NODATA_FOR_MISSING':False,'DATA_TYPE':5,'CREATE_OPTIONS':None,'OUTPUT':progressFolder + "/" + baseName + "reclass.gpkg"})

        #turning reclass raster into a raster layer object
        reclassRastRL = QgsRasterLayer(reclassRast['OUTPUT'], "reclassified raster", "gdal")

        #calculating zonal histogram from reclassified raster 
        zonalHistOP = progressFolder + "/" + baseName + "zonalHist.gpkg"
        zonalHist = processing.run("native:zonalhistogram", {'INPUT_RASTER':reclassRast['OUTPUT'],'RASTER_BAND':1,'INPUT_VECTOR':blocks,'COLUMN_PREFIX':'CLASS_','OUTPUT':zonalHistOP})

        #getting pixel size in order to get area
        pixel_size_x = reclassRastRL.rasterUnitsPerPixelX()
        pixel_size_y = reclassRastRL.rasterUnitsPerPixelY()
        pixelAreaFT = pixel_size_x * pixel_size_y
        pixelAreaAcres = pixelAreaFT / 43560

        #zonalHist transform to csv
        zonalHistCSV = progressFolder + "/" + baseName + "zonalHist_CSV"
        options = gdal.VectorTranslateOptions(format='CSV', layerCreationOptions=['GEOMETRY=AS_WKT'])
        gdal.VectorTranslate(zonalHistCSV, zonalHistOP, options=options)

        #renaming csv file that is created during the translation, because there is some kind of weird splitting happening in regard to the base name
        zonalHistCSV2 = zonalHistCSV + "/" + baseName + "zonalHist.csv"
        csvInFolder = os.listdir(zonalHistCSV)
        os.rename(zonalHistCSV + "/" + csvInFolder[0], zonalHistCSV2)

        #reading the csv
        df = pd.read_csv(zonalHistCSV2)

        histogramData = []
        fullAreas = []
        maxes = []

        row = 0

        for block in df['block']:
            counts = []
            classNum = 1
            blockAreaAcres = 0
            
            while (classNum <= 8):
                className = 'CLASS_' + str(classNum)
                #turnign pixel count into acres and adding it to a list to be used in the histogram
                counts.append(round(int(df.loc[row, className]) * pixelAreaAcres, 2))
            
                blockAreaAcres += (int(df.loc[row, className]) * pixelAreaAcres)

                classNum += 1
                
            histogramData.append(counts)
            fullAreas.append(round(blockAreaAcres, 2))
            maxes.append(max(counts))

            row += 1

        #I WANT TO MAKE IT SO THAT ALL OF THE OVERLAY OPTION HISTOGRAMS ARE ON ONE WINDOW
        subFigRows = math.ceil(len(histogramData))

        #setting up window to hold multiple histograms
        n_cols = 2
        n_rows = (len(histogramData) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.5   * n_rows))
        axes = axes.flatten()

        histNameNum = 0 
        histNames = df['block'].to_list()

        #bar colors
        #cmap = plt.colormaps['rainbow']
        #colors = [cmap(i / len(ranges)) for i in range(len(ranges))]
        colors = ['#a988ff', '#0600ff', '#00d9ff', '#00ff20', '#04ae00', '#fff900', '#ff834e', '#ce0000']
        
        print(len(ranges))
        print(len(colors))
        
        for i, data in enumerate(histogramData):
            if i < len(axes): # Ensure we don't try to plot on non-existent axes
                ax = axes[i]
                ax.bar(ranges, data, color = colors, edgecolor='black')
                ax.set_title(histNames[histNameNum])
                ax.set_xlabel('Depth to Water Table (ft)')
                ax.set_ylabel('Acres')
                ax.set_ylim(0, maxes[i] + (fullAreas[i] / 100))
                ax.ticklabel_format(axis='y', style='plain', scilimits=None, useOffset=None, useLocale=None, useMathText=None)
                histNameNum += 1

        plt.tight_layout()
        histogramWindowName = progressFolder + "/" + baseName + "hist.png"
        plt.suptitle(histPlotName)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(histogramWindowName)

        QgsMessageLog.logMessage(f"HISTOGRAM GENERATED: {histogramWindowName}", "Overlay Maker", Qgis.Info)


    #FLAT WATER TABLE GENERATOR: Creates a flat water table raster for specified blocks
    def flatWT(self, blocks, outputFolder):
        QgsMessageLog.logMessage("\n~ Performing flat water table generation ~", "Overlay Maker", Qgis.Info)

        #saving sources of clipped domes in order to merge later
        #flatWTList = []
        
        #for each selected layer (blocks)...
        baseName = os.path.basename(blocks).split(".")[0]
        
        #creating output paths for flat rasterized blocks and overlay
        outputPath = outputFolder  + '/flatRaster_' + baseName + '.tif'

        #create a flat raster based on the waterlevel of each blocks
        blockFlatRaster = processing.run("gdal:rasterize", {'INPUT':blocks,'FIELD':'wl','BURN':0,'USE_Z':False,'UNITS':1,'WIDTH':3,'HEIGHT':3,'EXTENT':None,'NODATA':0,'OPTIONS':None,'DATA_TYPE':5,'INIT':None,'INVERT':False,'EXTRA':'','OUTPUT':outputPath})

        QgsMessageLog.logMessage(">>> FLAT WATER TABLE GENERATED!: " + blockFlatRaster['OUTPUT'] + "\n", "Overlay Maker", Qgis.Info)

        return blockFlatRaster['OUTPUT']

    #DOMED WATER TABLE GENERATOR: Creates a domed water table raster for specified blocks
    def domedWT(self, domedBlocks = [], outputFolder = str, columnIndicator = 2, opt = int):
        QgsMessageLog.logMessage("\n~ Performing creation of domed water table: Option " + str(opt) + " ~", "Overlay Maker", Qgis.Info)
        
        #saving sources of clipped domes in order to merge later
        clippedDomes = []
            
        for dome in domedBlocks:
            #getting name of vector layer from path
            baseName = os.path.basename(dome).split(".")[0]
            
            #saving the vector in a variable
            domeLayer = QgsVectorLayer(dome, baseName, "ogr")
            
            #checking for validity of layer and skipping if necessary
            if not domeLayer.isValid():
                print("\n" + baseName + " is not valid. \nSkipping...\n")
                continue
            
            #getting extent of domed block 
            extent = domeLayer.extent()
            
            #setting up path and instructions for TIN interpolation tool
            interpolationData = dome + '|layername=' + baseName +'::~::0::~::' + str(columnIndicator) + '::~::2'

            QgsMessageLog.logMessage("\n~WE MAKE IT HERE IN DOMED WT " + str(opt) + " ~", "Overlay Maker", Qgis.Info)

            #calculating rough dome and adding to map viewer
            if outputFolder == 'TEMPORARY_OUTPUT':
                roughOutput = 'TEMPORARY_OUTPUT'
                clippedOutput = 'TEMPORARY_OUTPUT'
            else:
                roughOutput = outputFolder + "/" + baseName + "_domeRough_" + str(opt) 
                clippedOutput = outputFolder + "/" + baseName + "_domeClipped_" + str(opt) 
            
            domeRough = processing.run("qgis:tininterpolation", {'INTERPOLATION_DATA':interpolationData,'METHOD':0,'EXTENT':extent,'PIXEL_SIZE':15,'OUTPUT': roughOutput + ".tif"})
            #print("Rough dome created: " + domeRough['OUTPUT'])

            QgsMessageLog.logMessage(f"\nClip by extent inputs: domeRough = {domeRough['OUTPUT']}, output: {clippedOutput}", "Overlay Maker", Qgis.Info)

            #clipping dome to block
            domeClipped = processing.run("gdal:cliprasterbyextent", {'INPUT':domeRough['OUTPUT'],'PROJWIN': extent,'OVERCRS':False,'NODATA':None,'OPTIONS':None,'DATA_TYPE':0,'EXTRA':'','OUTPUT':clippedOutput + '.tif'})
            

            QgsMessageLog.logMessage(f"\n~Clipped Dome Created: {clippedOutput} ", "Overlay Maker", Qgis.Info)
            self.app.dlg.progressUpdates.append(f"\n~Clipped Dome Created: {clippedOutput} ")

            #adding final clipped dome to a list
            clippedDomes.append(domeClipped['OUTPUT'])


        #merging all domes 
        if outputFolder == 'TEMPORARY_OUTPUT':
            merge = processing.run("gdal:merge", {'INPUT':clippedDomes,'PCT':False,'SEPARATE':False,'NODATA_INPUT':None,'NODATA_OUTPUT':0,'OPTIONS':None,'EXTRA':'','DATA_TYPE':5,'OUTPUT': 'TEMPORARY_OUTPUT'})
            QgsMessageLog.logMessage("COMPLETED DOME: " + merge['OUTPUT'] + "\n", "Overlay Maker", Qgis.Info)
            return merge
        else:
            QgsMessageLog.logMessage(f"\nMerge inputs: domeRough = {clippedDomes}", "Overlay Maker", Qgis.Info)

            merge = processing.run("gdal:merge", {'INPUT':clippedDomes,'PCT':False,'SEPARATE':False,'NODATA_INPUT':None,'NODATA_OUTPUT':0,'OPTIONS':None,'EXTRA':'','DATA_TYPE':5,'OUTPUT':outputFolder + '/all_domedWT_merged_' + str(opt) + "ft.tif"})
            QgsMessageLog.logMessage("COMPLETED DOME: " + merge['OUTPUT'] + "\n", "Overlay Maker", Qgis.Info)

            return merge

    #ROAD CALC: Creates a vector layer showing roads in the project area in need of potential raising

    def roadRaisingLength(self, roads, overlay, outputFolder):
        roadOutputPath = self.existingFolderHandler(outputFolder, "Roads")

        #takes in project overlay
        overlayRL = QgsRasterLayer(overlay, "Overlay", "gdal")
    
        #Takes in line vector of middle of roads in project area
        roadsVL = QgsVectorLayer(roads, "Roads", "ogr")
        #buffers it by +15
        bufferedRoads = processing.run("native:buffer", {
        'INPUT': roadsVL.source(),
        'DISTANCE': 15,
        'SEGMENTS': 5,
        'END_CAP_STYLE': 0,  # Round end cap style
        'JOIN_STYLE': 1,     # Round join style
        'MITER_LIMIT': 2,
        'DISSOLVE': False,   # Set to True to dissolve all buffers
        'OUTPUT': 'memory:'  # Create a temporary layer

    })
        #clips overlay to road buffer
        clippedOverlay = processing.run("gdal:cliprasterbymasklayer", {'INPUT':overlayRL.source(),'MASK':bufferedRoads['OUTPUT'],'SOURCE_CRS':None,'TARGET_CRS':None,'TARGET_EXTENT':None,'NODATA':None,'ALPHA_BAND':False,'CROP_TO_CUTLINE':True,'KEEP_RESOLUTION':False,'SET_RESOLUTION':False,'X_RESOLUTION':None,'Y_RESOLUTION':None,'MULTITHREADING':False,'OPTIONS':None,'DATA_TYPE':0,'EXTRA':'','OUTPUT':'TEMPORARY_OUTPUT'})
        
        #reclassifies road overlay raster
        reclassRoadRast = processing.run("native:reclassifybytable", {'INPUT_RASTER':clippedOverlay['OUTPUT'],'RASTER_BAND':1,'TABLE':['-inf','0','1','0','1','2','1','2','3','2','inf','4'],'NO_DATA':-9999,'RANGE_BOUNDARIES':0,'NODATA_FOR_MISSING':False,'DATA_TYPE':5,'CREATE_OPTIONS':None,'OUTPUT':'TEMPORARY_OUTPUT'})

        #vectorize road overlay raster
        vectorizedRoad = processing.run("native:pixelstopolygons", {'INPUT_RASTER':reclassRoadRast['OUTPUT'],'RASTER_BAND':1,'FIELD_NAME':'VALUE','OUTPUT':'TEMPORARY_OUTPUT'})

        #calc intersection w road lines -> will create anothr set of lines
        roadIntersect = processing.run("native:intersection", {'INPUT':roadsVL.source(),'OVERLAY':vectorizedRoad['OUTPUT'],'INPUT_FIELDS':[],'OVERLAY_FIELDS':['VALUE'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})

        #dissolve vectorized road overlay raster based on classification
        #make permanent
        dissolvedRoadVector = processing.run("native:dissolve", {'INPUT':roadIntersect['OUTPUT'],'FIELD':['VALUE'],'SEPARATE_DISJOINT':False,'OUTPUT':roadOutputPath + "/Road Vector.shp"})

        #calculate length of each feature (length($geometry) is the accurate one) in the final road vector in miles
        layer = QgsVectorLayer(dissolvedRoadVector['OUTPUT'], "dissolvedVectorLayer", "ogr")

        # Ensure the layer is in editing mode
        if not layer.isEditable():
            layer.startEditing()

        # Add a new field for the length
        # The type QVariant.Double is suitable for decimal numbers
        res = layer.dataProvider().addAttributes([QgsField("miles", QVariant.Double, 'Double', 10, 2)])
        layer.updateFields()

        # Get the index of the newly created field
        fieldIndex =  layer.fields().indexOf("miles")

        # Calculate the length for each feature and update the attribute table
        for feature in layer.getFeatures():
            geom = feature.geometry()
            # geom.length() calculates planar length in the layer's CRS units
            length = round(geom.length() / 5280, 2)
            layer.changeAttributeValue(feature.id(), fieldIndex, length)

        # Commit changes and stop editing
        layer.updateFields()
        layer.commitChanges()
            

        options = gdal.VectorTranslateOptions(format='CSV')
        csvOutputPath = roadOutputPath + "/roadStats"
        os.mkdir(csvOutputPath)
        gdal.VectorTranslate(csvOutputPath + "/CSV", layer.source(), options=options)

        QgsMessageLog.logMessage(f"Road raising CSV: {csvOutputPath}" , "Overlay Maker", Qgis.Info)



    def roadFillCalc(self, dem, roads, WT, outputFolder):

        #takes in geometry of 
        QgsMessageLog.logMessage("\n~ Performing calculation of affected roads in project area ~\n", "Overlay Maker", Qgis.Info)

        #getting basename of file being used
        baseName = os.path.basename(roads).split(".")[0] + "_overlay"
            
        outputPath = outputFolder + "/" + baseName

        roadRasterOP = "'" + outputFolder + "/" + "roadRaster.tif'"

        #buffering the roads vector

        #clipping dem to roads
        roadsRaster = processing.run("gdal:cliprasterbymasklayer", {'INPUT':dem,'MASK':roads,'SOURCE_CRS':None,'TARGET_CRS':None,'TARGET_EXTENT':None,'NODATA':None,'ALPHA_BAND':False,'CROP_TO_CUTLINE':True,'KEEP_RESOLUTION':False,'SET_RESOLUTION':False,'X_RESOLUTION':None,'Y_RESOLUTION':None,'MULTITHREADING':False,'OPTIONS':None,'DATA_TYPE':0,'EXTRA':'','OUTPUT':'C:/wfh/python/overlayMaker/RR'})


        #subtracting proposed water table from rasterized version of the roads
        #roadsOverlay = rasterSubtractor.rasterSubtractor(roadsRaster['OUTPUT'], WT, outputPath)

    def autoOverlay(self, task, blocks, dem, outputFolder, overlayOptions = []):

        print("\n------------------------------------------------------------------------------------------------------------------------------------------")
        print("\n---------------------------------------------STARTING OVERLAYMAKER------------------------------------------------------------------------")
        print("\n------------------------------------------------------------------------------------------------------------------------------------------")

        #----------testing inputs begin----------

        QgsMessageLog.logMessage(f"Blocks: {blocks}", "Overlay Maker", Qgis.Info)
        self.app.dlg.progressUpdates.append(f"\nBlocks: {blocks}")


        testFunctions.hasRequiredColumnTest(blocks, 'block')
        wlColumnPresent = testFunctions.hasOptionalColumnTest(blocks, 'wl')

        #----------testing inputs end----------
        
        #-----------FOLDER CREATION---------------------------------------------------

        #-----------OUTPUT FOLDER------------
        autoOverlayFolder = outputFolder + "/autoOverlay"

        if not os.path.exists(autoOverlayFolder):
            os.makedirs(autoOverlayFolder)
            outputFolder = autoOverlayFolder
        else:
            # Folder already exists, find a unique name by appending a number
            folder_name, _ = os.path.splitext(autoOverlayFolder) # Split to handle potential extensions, though not typical for folders
            
            counter = 1

            folderExists = True

            while folderExists == True:

                new_folder_path = f"{folder_name} ({counter})"

                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)

                    outputFolder = new_folder_path

                    folderExists = False

                else:
                    counter += 1

        #------------PROCESSING--------------------
        #Making Processing folder
        ProcessingFolder = outputFolder + "/Processing"
        os.mkdir(ProcessingFolder)

        #--------------PROCESSING -> DEM STATS-----------------
        demStatsFolder = ProcessingFolder + "/Initial Block DEM Stats"
        os.mkdir(demStatsFolder)

        #--------------PROCESSING -> TOD-----------------
        #Making TOD folder
        TODFolder = ProcessingFolder + "/TOD"
        os.mkdir(TODFolder)

        #Making TOD Stats folder
        TODStatsFolder = TODFolder + "/TOD Stats"
        os.mkdir(TODStatsFolder)

        #Making TOD Vectors folder
        TODVectorsFolder = TODFolder + "/TOD Vectors"
        os.mkdir(TODVectorsFolder)

        #-------------PROCESSING -> DOMES-------------------

        #Making dome folder
        domesFolder = ProcessingFolder + "/Domes"
        os.mkdir(domesFolder)

        #Making merged dome vector folder
        blocksInnerOuter = domesFolder + "/Merged Dome Vectors"
        os.mkdir(blocksInnerOuter)

        #Making domed water table raster folder
        domedWTOutputPath = domesFolder + "/Domed Water Tables"
        os.mkdir(domedWTOutputPath)

        #-------------PROCESSING -> HISTOGRAMS-------------------

        #Making histogram progress folder
        histogramsProgFolder = ProcessingFolder + "/Histograms"
        os.mkdir(histogramsProgFolder)

        #-------------OVERLAYS-------------------
        #Making Completed Overlays folder
        overlaysFolder = outputFolder + "/Completed Overlays"
        os.mkdir(overlaysFolder)

        #-------------HISTOGRAMS-------------------
        histogramsFolder = outputFolder + "/Completed Histograms"
        os.mkdir(histogramsFolder)

        #-------------ESTABLISHING IMPORTANT VARIABLES-------------------

        #adding other wl columns for overlay options, if not already defined
        if len(overlayOptions) == 0:
            QgsMessageLog.logMessage("Overlay options not specified. Using default options: [0, 1, 2, -1, -2].", "Overlay Maker", Qgis.Info)
            overlayOptions = [0]

        #variable for getting the index of the wl columns so we can have the proper TIN interp settings later
        wlIndexes = []

        #LIST FOR HOLDING MERGED BLOCKS WITH INNER AND OUTER BOUNDARIES
        domeBlocks = []

        #calculating DEM stats: count, sum, mean, stdv
        demStats = processing.run("native:zonalstatisticsfb", {'INPUT':blocks,'INPUT_RASTER':dem,'RASTER_BAND':1,'COLUMN_PREFIX':'_','STATISTICS':[0,1,2,4],'OUTPUT':demStatsFolder + "/Inital DEM Stats"})
        
        print("\n---------------------------------------------INITIAL BLOCK ANALYSIS------------------------------------------------------------------------")
        print("\n-> Initial block DEM Stats Created: " + "'" + demStats['OUTPUT'] + "'\n")

        #creating vector layer for demStats file
        demStatsVL = QgsVectorLayer(demStats['OUTPUT'], "DEMStats")
        
        #if the wl column is not already present in the layer:
        if wlColumnPresent == False:
            #creating a wl field for the blocks
            blockWLs = QgsField("wl", QMetaType.Double) 

            #opening editor
            with edit(demStatsVL):
                #getting data provider
                demStatsVLpr = demStatsVL.dataProvider()
                #adding wl
                demStatsVLpr.addAttributes([blockWLs])
                demStatsVL.updateFields()

                #adding wl (mean - 1stdv)
                for feature in demStatsVL.getFeatures():
                    feature.setAttribute(feature.fieldNameIndex('wl'), round(round((feature['_mean'] - feature['_stdev']) * 2) / 2, 2))
                    demStatsVL.updateFeature(feature)

        #splitting each block into its own layer
        splitBlocksInput = demStats["OUTPUT"] 
        splitBlocks = processing.run("native:splitvectorlayer", {'INPUT':splitBlocksInput,'FIELD':'block','PREFIX_FIELD':True,'FILE_TYPE':1,'OUTPUT':ProcessingFolder + "/All Blocks Split"})
        
        QgsMessageLog.logMessage("--> Blocks split: " + "'" + splitBlocks['OUTPUT'], "Overlay Maker", Qgis.Info)
        self.app.dlg.progressUpdates.append("\n --> Blocks split: '" + splitBlocks['OUTPUT'] + "'")



        #making grid for each block to determine highest point and add this as a dome feature to the block
        wlColIndexCollected = False
        for b in splitBlocks['OUTPUT_LAYERS']:
            #Getting base name of current block 
            currentBlock = os.path.basename(b).split(".")[0] 

            QgsMessageLog.logMessage("\n>>>>>>>>>>>>>>> Analyzing " + currentBlock + " <<<<<<<<<<<<<<<", "Overlay Maker", Qgis.Info)
            
            #turning block path into a vector layer
            block = QgsVectorLayer(b, "block", "ogr")

            #getting area of current block
            blockArea = 0
            
            #getting area of each feature in layer and adding up
            for feature in block.getFeatures():
                geom = feature.geometry()
                blockArea += geom.area()

            #getting area in acres
            blockAreaAcres = round(blockArea / 43560, 1)

            QgsMessageLog.logMessage(f"\n-> Area of {currentBlock} = " + str(blockAreaAcres) + " acres", "Overlay Maker", Qgis.Info)

            #creating grid 
            grid = processing.run("native:creategrid", {'TYPE':0,'EXTENT':block.extent(),'HSPACING':610,'VSPACING':610,'HOVERLAY':0,'VOVERLAY':0,'CRS':QgsCoordinateReferenceSystem(block.crs()),'OUTPUT':'TEMPORARY_OUTPUT'})

            #clipping grid to block 
            clippedGrid = processing.run("native:clip", {'INPUT':grid['OUTPUT'],'OVERLAY':block,'OUTPUT':'TEMPORARY_OUTPUT'})
        
            #buffering the grid
            buffered = processing.run("native:buffer", {'INPUT':clippedGrid['OUTPUT'],'DISTANCE':(blockArea/10000),'SEGMENTS':5,'END_CAP_STYLE':0,'JOIN_STYLE':0,'MITER_LIMIT':2,'DISSOLVE':False,'SEPARATE_DISJOINT':False,'OUTPUT':'TEMPORARY_OUTPUT'})
            
            #clipping buffer to block 
            clippedBuffer = processing.run("native:clip", {'INPUT':buffered['OUTPUT'],'OVERLAY':block,'OUTPUT':'TEMPORARY_OUTPUT'})

            #zonal stats on the grid vectors
            TODStats = processing.run("native:zonalstatisticsfb", {'INPUT':clippedBuffer['OUTPUT'],'INPUT_RASTER':dem,'RASTER_BAND':1,'COLUMN_PREFIX':'_','STATISTICS':[0,1,2,4],'OUTPUT':TODStatsFolder + "/" + currentBlock + "_TODStats"})
            
            print("--> Block grid created: " + "'" + TODStats['OUTPUT'])

            QgsMessageLog.logMessage("--> Block grid created: " + "'" + TODStats['OUTPUT'], "Overlay Maker", Qgis.Info)

            #Making this into a vector layer
            TODStatsVL = QgsVectorLayer(TODStats['OUTPUT'], "TODStatsVL_" + currentBlock, "ogr")
            
            #adding a wl column for easy merging
            wl = QgsField("wl", QMetaType.Double) 

            #editing vector layer
            TODStatsVL.startEditing()

            #using data provider to add wl column
            TODStatsVLpr = TODStatsVL.dataProvider()
            TODStatsVLpr.addAttributes([wl])

            #closing out editing and saving
            TODStatsVL.updateFields()
            TODStatsVL.commitChanges()

            #getting index of max mean value in attribute table
            meanIndex = TODStatsVL.fields().indexFromName("_mean")
            #getting max mean 
            maxMean = TODStatsVL.maximumValue(meanIndex)

            #getting feature id of max mean
            found_feature_id = None
            for feature in TODStatsVL.getFeatures():
                if feature['_mean'] == maxMean:
                    found_feature_id = feature.id()
                    break
            
        #selecting dome with highest mean
            TODStatsVL.selectByIds([found_feature_id])

            #creating a new vector file for dome top
            if TODStatsVL.selectedFeatureCount() > 0:
            
                save_options = QgsVectorFileWriter.SaveVectorOptions()
                save_options.driverName = "ESRI Shapefile"
                save_options.layerName = "new_layer_name"
                save_options.onlySelectedFeatures = True  # Set to True to export only selected features
                save_options.destCRS = TODStatsVL.crs() # Export with the same CRS
                save_options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile
                transform_context = QgsProject.instance().transformContext()

                topOfDome = QgsVectorFileWriter.writeAsVectorFormatV3(layer=TODStatsVL, fileName=TODVectorsFolder + "/" + currentBlock + "_TOD_isolated", transformContext=transform_context, options=save_options)

                QgsMessageLog.logMessage("---> Dome selected for " + currentBlock, "Overlay Maker", Qgis.Info)

            else:
                QgsMessageLog.logMessage("\n---> ERROR: No features selected from grid\n", "Overlay Maker", Qgis.Info)

            
            
            #merging TOD with block        
            domeBlockMerged = processing.run("native:mergevectorlayers", {'LAYERS':[b, topOfDome[2]],'CRS':None,'OUTPUT':blocksInnerOuter + "/" + currentBlock + "_inner+outer"})
            QgsMessageLog.logMessage("----> Merged top of dome with block: " + "'" + domeBlockMerged['OUTPUT'], "Overlay Maker", Qgis.Info)


            #making a vector layer from the merged path
            domeBlockMergedVL = QgsVectorLayer(domeBlockMerged['OUTPUT'], "domeBlockMergedVL", "ogr")

            #make sure to include the base overlay option 
            attribute = domeBlockMergedVL.fields().indexOf('wl')
            blockAttrIndex = domeBlockMergedVL.fields().indexOf('block')
        
            #editing vector layer
            domeBlockMergedVL.startEditing()

            #creating data provider to add wl data 
            domeBlockMergedVLpr = domeBlockMergedVL.dataProvider()

            innerID = 0
            TODwL = 0
            blockName = ''

            if wlColumnPresent == True:
                for feature in domeBlockMergedVL.getFeatures():
                    if QgsVariantUtils.isNull(feature['wl']) == False:
                        TODwL = feature['wl'] + feature['_stdev']
                        blockName = feature['block']
                    else:
                        innerID = feature.id()
            else:
                for feature in domeBlockMergedVL.getFeatures():
                    if QgsVariantUtils.isNull(feature['wl']) == False:
                        TODwL = feature['_mean']
                        blockName = feature['block']
                    else:
                        innerID = feature.id()

            #setting top of dome water level to mean 
            domeBlockMergedVL.changeAttributeValue(innerID, attribute, TODwL)
            #changing inner block name to match overall block name
            domeBlockMergedVL.changeAttributeValue(innerID, blockAttrIndex, blockName + "_inner")
            
            domeBlockMergedVL.updateFields()
            
            #adds in the column then fills
            for overlay in overlayOptions:
                wlFieldName = "wl_" + str(overlay)

                wlField = QgsField(wlFieldName, QMetaType.Double) 

                domeBlockMergedVLpr.addAttributes([wlField])
                domeBlockMergedVL.updateFields()            
            
                #and fills with corresponding overlay calc 
                for feature in domeBlockMergedVL.getFeatures():
        
                    feature.setAttribute(feature.fieldNameIndex(wlFieldName), float(feature['wl']) + overlay)
                    
                    domeBlockMergedVL.updateFeature(feature)
            
                if wlColIndexCollected == False:
                    wlIndexes.append(feature.fieldNameIndex(wlFieldName))
            
            wlColIndexCollected = True

            domeBlockMergedVL.commitChanges()
            domeBlocks.append(domeBlockMerged['OUTPUT'])
            
            QgsMessageLog.logMessage("-----> Initial block dome vector created and wls calculated: " + "'" + domeBlockMerged['OUTPUT'], "Overlay Maker", Qgis.Info)


        
        #going through every option 
        
        overlayOptionIndex = 0

        QgsMessageLog.logMessage("\n--------------------------CREATING DOMES AND OVERYLAYS--------------------------\n", "Overlay Maker", Qgis.Info)

        domedWaterTable = None

        for index in wlIndexes:

            #creating domes for each overlay option 
            QgsMessageLog.logMessage(f"\n>>>>>>>>>>>>>>> CREATING DOMES, OVERLAY AND HISTOGRAM FOR {overlayOptions[overlayOptionIndex]} ft OVERLAY OPTION <<<<<<<<<<<<<<<", "Overlay Maker", Qgis.Info)
            QgsMessageLog.logMessage("We make it here before crashing!", "Overlay Maker", Qgis.Info)

            QgsMessageLog.logMessage(f"\nCurrent Inputs: domeBlocks = {domeBlocks}, domeWTOutputPath = {domedWTOutputPath}", "Overlay Maker", Qgis.Info)

            domedWaterTable = self.domedWT(domeBlocks, domedWTOutputPath, index, overlayOptions[overlayOptionIndex])

            QgsMessageLog.logMessage("AND NOW WERE HERE", "Overlay Maker", Qgis.Info)
            

            #resampling dome to match dem so it can successfully subtract it
            resampledDomeOutput = domedWTOutputPath + "/" + str(overlayOptions[overlayOptionIndex]) + "_resampled"
            DEMbaseName = os.path.basename(dem).split(".")[0]

            resampledDEMOutput = domedWTOutputPath + "/" + DEMbaseName + "_resampled"
            resamp = processing.run("native:alignrasters", {'LAYERS':[{'inputFile': domedWaterTable['OUTPUT'],'outputFile': resampledDomeOutput,'resampleMethod': 0,'rescale': False},{'inputFile': dem,'outputFile': resampledDEMOutput + ".tif",'resampleMethod': 0,'rescale': False}],'REFERENCE_LAYER':dem,'CRS':None,'CELL_SIZE_X':None,'CELL_SIZE_Y':None,'GRID_OFFSET_X':None,'GRID_OFFSET_Y':None,'EXTENT':dem})

            QgsMessageLog.logMessage("-> Dome resampled to match DEM", "Overlay Maker", Qgis.Info)

            #creating overlay  
            overlay = self.rasterSubtractor(dem, resampledDomeOutput, overlaysFolder, overlayOptions[overlayOptionIndex])
            
            #clipping overlay
            blockBoundaries = QgsVectorLayer(blocks, "blocks", "ogr")
            overlayRasterSource = QgsRasterLayer(overlay, "Clipped Overlay " + str(overlayOptions[overlayOptionIndex]) , "gdal").source()
            clippedOverlayPath = overlaysFolder + "/Clipped Overlay " + str(overlayOptions[overlayOptionIndex]) + ".tif"

            params = {'INPUT':overlay,
                'MASK':blockBoundaries.source(),
                'SOURCE_CRS':None,
                'TARGET_CRS':None,
                'TARGET_EXTENT':None,
                'NODATA':None,
                'ALPHA_BAND':False,
                'CROP_TO_CUTLINE':True,
                'KEEP_RESOLUTION':False,
                'SET_RESOLUTION':False,
                'X_RESOLUTION':None,
                'Y_RESOLUTION':None,
                'MULTITHREADING':False,
                'OPTIONS':'',
                'DATA_TYPE':0,
                'EXTRA':'-co COMPRESS=LZW',
                'OUTPUT':clippedOverlayPath}
            
            clippedOverlay = processing.run("gdal:cliprasterbymasklayer", params)

            QgsMessageLog.logMessage(f"\n -> CLIPPED OVERLAY GENERATED: '{clippedOverlay['OUTPUT']}'", "Overlay Maker", Qgis.Info)

            #creating histogram
            self.rasterHist(overlay, blocks, histogramsProgFolder, histogramsFolder, None, f"Overlay Option: {overlayOptions[overlayOptionIndex]} ft")
        
            overlayOptionIndex += 1
        
        print("\n-------------------------- AUTO-OVERLAY COMPLETE :) --------------------------\n")
        QgsMessageLog.logMessage("\n-------------------------- AUTO-OVERLAY COMPLETE :) --------------------------\n", "Overlay Maker", Qgis.Info)


