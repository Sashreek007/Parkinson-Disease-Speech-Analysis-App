import reflex as rx
from ..models.parkinson import *

class ParkState(rx.State):
    """The app state."""
    prediction: str = ""  # Store the prediction result

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        if not files:
        # no files or too many files selected
            return
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            # Save the file
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            # Process the audio file and make prediction
            try:
                features = extract_features(str(outfile))
                features = features.reshape(1, -1)
                prediction = best_rf_model.predict(features)
                self.prediction = "Healthy" if prediction == 1 else "Not Healthy"
            except Exception as e:
                self.prediction = f"Error processing file: {str(e)}"
