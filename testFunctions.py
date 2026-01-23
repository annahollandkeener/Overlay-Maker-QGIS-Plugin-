#test functions
import os
from qgis.core import QgsVectorLayer, QgsVariantUtils, QgsMessageLog, Qgis

def isPathValidTest(pathToTest):
    #check if valid path
    if not os.path.exists(pathToTest):
          print("ERROR: " + pathToTest + " is not a valid path.")
          print("Make sure your backslashes have been forwarded!")
          raise FileNotFoundError
    
#check if the attribute table has a wl column
def hasOptionalColumnTest(blocksToTest = str, colName = str):
    #making input block a vector layer in order to access attribute table
    blocksVL = QgsVectorLayer(blocksToTest)

    found = False

    QgsMessageLog.logMessage(f"Column name:{colName}", "Overlay Maker", Qgis.Info)

    #checking for column
    for field in blocksVL.fields():
        if field.name() == colName:
            found = True
            break

    #if the 'wl field doesn't exist, the index will be = -1  
    if found == False:
        print(f"The field '" + colName + f"' does not exist in layer '{blocksToTest}'.")
        print("---> Creating new '" + colName + "'  column based on DEM for water table generation.")
        
        return False
      
    else:
        print("\nHEADS UP!: The field '" + colName + f"' exists in layer '{blocksToTest}'. Using existing field for water table generation.")
        
        return True
    
def hasRequiredColumnTest(blocksToTest, colName):
    #making input block a vector layer in order to access attribute table
    blocksVL = QgsVectorLayer(blocksToTest)

    QgsMessageLog.logMessage(f"Blocks to test:{blocksVL}", "Overlay Maker", Qgis.Info)

    QgsMessageLog.logMessage(f"Column name:{colName}", "Overlay Maker", Qgis.Info)

    
    found = False

    #checking for column
    for field in blocksVL.fields():
        QgsMessageLog.logMessage(f"Field: {field}", "Overlay Maker", Qgis.Info)

        if field.name() == colName:
            found = True
            break

    #if the field doesn't exist, the index will be = -1  
    if found == False:
        raise Exception(("The field '" + colName + f"' does not exist in layer '{blocksToTest}'. Add this field and associated features in the attribute table."))
    

'''
    for feature in blocksVL.getFeatures():
        field_value = feature(colName)

        if QgsVariantUtils.isNull(field_value):  
            raise Exception((f"ERROR: There are null values in {colName}"))

'''

        
        

#need to check that the blocks input has a block column 
