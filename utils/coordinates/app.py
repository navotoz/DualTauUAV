from pathlib import Path
from flask import Flask, render_template, request

from tempfile import NamedTemporaryFile

from utils_coord import process_dgps_pts
SLEEP_GENERATOR_SEC = 0.5

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize the results to None
    height = ""
    height_takeoff = ""
    heights = [""]
    latitude = ""
    longitude = ""
    north = ""
    east = ""
    results = ""

    # Check if the request method is POST
    if request.method == "POST":
        # Get the values from the input fields
        height_takeoff = request.form.get("height-takeoff")
        height = request.form.get("height")
        north = request.form.get("north")
        east = request.form.get("east")

        with NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as file_csv:
            file_csv.write(f'PRS748093119329,669767.553,183795.761,43.691,\n1,{north},{east},{height},')

        coordinates, heights = process_dgps_pts(file_csv.name,
                                                skip_rows_list=[0],  # usually the first row is a setup point
                                                height_takeoff=float(height_takeoff),
                                                heights_list=[12, 48, 96],
                                                to_csv=False)

        Path(file_csv.name).unlink()

        # Concatenate the values with a comma
        longitude = str(coordinates[1][1])
        latitude = str(coordinates[1][2])

    # Render a template with three input fields and a hidden div for the results
    return render_template("index.html", results=results, height=height, north=north, east=east,
                           height_takeoff=height_takeoff, height_list=str(heights[0]),
                           latitude=latitude, longitude=longitude)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
