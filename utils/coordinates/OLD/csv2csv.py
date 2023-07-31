import argparse
import os
import csv
import pyproj


def process_csv(input_file):
    # Read the input CSV
    name=os.path.basename(input_file)
    transformer = pyproj.Transformer.from_crs("EPSG:6991", 4326)
    write_rows=[]
    write_height=[]
    with open(input_file, 'r') as csv_in_file:
        reader = csv.reader(csv_in_file)

        line_count=0
        for i,row in enumerate(reader):
            print(i,row)
            if line_count==0: #skip first row rtk csv contains north tel aviv as first row
                line_count += 1
                write_rows.append(['point_name','lon','lat','height','heading','gimbal','speed','turnmode','actions_sequence'])
                continue
            if line_count==1:
                initial_height=row[3] #initial height is the height of take off
                line_count+=1
            east=float(row[2])
            north=float(row[1])
            coords=transformer.transform(east,north)

            write_rows.append([row[0],coords[1],coords[0],float(row[3])-float(initial_height),180,0,4,'C',''])
            write_height.append([float(row[3])-float(initial_height)+12,float(row[3])-float(initial_height)+48,float(row[3])-float(initial_height)+96])
            # Replace the following lines with your specific data processing logic
    print(write_rows)

    with open(name.split('.')[0]+'_out.csv', 'w', newline='') as csv_out_file:

        writer = csv.writer(csv_out_file)
        writer.writerows(write_rows)

    with open('heights.csv','w',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(write_height)


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process a CSV file with esgp:6991 coordinates and generate a new CSV file with wsg84 coords.")
    parser.add_argument("-i","--input_file",required=True,type=str, help="Input CSV filename to be processed")

    args = parser.parse_args()

    print(args.input_file)
    process_csv(args.input_file)

