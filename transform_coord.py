import argparse

from utils.utils_coord import coord_to_dji_kml, process_dgps_pts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a CSV file with esgp:6991 coordinates and generate a new CSV file with wsg84 coords.")
    parser.add_argument("-i", "--input_file", required=True, type=str, help="Input CSV filename to be processed")

    args = parser.parse_args()

    print("\nCoordinates file from DGPS to transform:", args.input_file)
    coordinates, heights = process_dgps_pts(args.input_file,
                                            skip_rows_list=[0],  # usually the first row is a setup point
                                            height_takeoff=20,
                                            heights_list=[12, 48, 96], to_csv=True)
    print('\nHeights:')
    [print(i, h) for i, h in enumerate(heights, start=int(coordinates[1][0]))]
    print('\nDJI coordinates:')
    coord_to_dji_kml(coordinates=coordinates)
