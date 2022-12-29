from fastapi import FastAPI, File, UploadFile
from enum import Enum
from fastapi.responses import FileResponse
import PBNifier

# initialize API server
app = FastAPI(title="PBNify", description="Painting-by-number generation from image")


# create custom class for path parameter validation
class Result(str, Enum):
    CANVAS = "canvas"
    COLORED = "colored"
    PALETTE = "palette"


# UploadFile class is used to input image file
# FileResponse class is used to get image response
@app.get("/{result_type}", response_class=FileResponse)
async def show_result(result_type: Result,
                      file: UploadFile = File(...)) -> FileResponse:
    painting = PBNifier.Painting(file, nb_colors=3, pixel_size=1000, save=False)
    painting.generate()
    result_paths = painting.get_paths()
    if result_type == "canvas":
        result_path = result_paths[0]
    elif result_type == "colored":
        result_path = result_paths[1]
    else:
        result_path = result_paths[2]
    return result_path
