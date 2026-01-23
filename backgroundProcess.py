from qgis.core import QgsApplication, QgsTask, QgsMessageLog

from . import model 

class CustomBackgroundProcess(QgsTask):
    def __init__(self, description, additional_data, m):
        super().__init__(description, QgsTask.CanCancel)
        self.additional_data = additional_data
        self.processing_result = None
        self.model = m

    def run(self):
        """Called in the background thread."""
        try:
            if self.additional_data['name'] == "AO":
                self.model.autoOverlay(self.additional_data['blocks'], self.additional_data['dem'], self.additional_data['output'])
            return True
        
        except Exception as e:
            self.exception = e
            return False