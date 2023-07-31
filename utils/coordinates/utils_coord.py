from string import Template
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union
import pyproj
ISRAELI_GRID = "EPSG:6991"
WGS84_GRID = 4326


def process_dgps_pts(input_file: Union[str, Path],
                     height_takeoff: float,
                     skip_rows_list: list = [0],
                     heights_list: list = [12, 48, 96], to_csv: bool = False):
    # Read the input CSV
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist")
    transformer = pyproj.Transformer.from_crs(ISRAELI_GRID, WGS84_GRID)
    write_rows = []
    write_rows.append(['point_name', 'lon', 'lat', 'height', 'heading',
                      'gimbal', 'speed', 'turnmode', 'actions_sequence'])
    write_height = []
    with open(input_file, 'r') as csv_in_file:
        reader = csv.reader(csv_in_file)

        for i, row in enumerate(reader):
            print(i, row)
            if i in skip_rows_list:  # skip first row rtk csv contains north tel aviv as first row
                continue
            east = float(row[2])
            north = float(row[1])
            coords = transformer.transform(east, north)

            write_rows.append([row[0],
                               coords[1],
                               coords[0],
                               float(row[3])-float(height_takeoff),
                               180, 0, 4, 'C', ''])
            write_height_list = []
            for height in heights_list:
                write_height_list.append(float(row[3])-float(height_takeoff)+height)
            write_height.append(write_height_list)

    if to_csv:
        with open(path.stem + '_out.csv', 'w', newline='') as csv_out_file:
            writer = csv.writer(csv_out_file)
            writer.writerows(write_rows)

        with open(path.stem + '_heights.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(write_height)
    return write_rows, write_height


def coord_to_dji_kml(coordinates: list, path_output_file: str = 'pilot.kml', on_finish: str = 'hover'):
    if on_finish.lower() == 'hover':
        on_finish = "Hover"
    elif on_finish.lower() == 'gohome':
        on_finish = "GoHome"
    else:
        raise ValueError(f"on_finish must be hover or gohome, not {on_finish}")

    CSV_HEADER = False

    XML_string = """<?xml version="1.0" encoding="UTF-8"?>

  <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document xmlns="">
      <name>chambon_small</name>
      <open>1</open>
      <ExtendedData xmlns:mis="www.dji.com">
        <mis:type>Waypoint</mis:type>
        <mis:stationType>0</mis:stationType>
      </ExtendedData>
      <Style id="waylineGreenPoly">
        <LineStyle>
          <color>FF0AEE8B</color>
          <width>6</width>
        </LineStyle>
      </Style>
      <Style id="waypointStyle">
        <IconStyle>
          <Icon>
            <href>https://cdnen.dji-flighthub.com/static/app/images/point.png</href>
          </Icon>
        </IconStyle>
      </Style>
      <Folder>
        <name>Waypoints</name>
        <description>Waypoints in the Mission.</description>\n"""
    # name = None
    # lon = None
    # lat = None
    # height = None
    # heading = None
    # gimbal = None
    all_coordinates = ""
    waypoint_number = 1

    waypoint_start = Template("""      <Placemark>
          <name>Waypoint$waypoint_number</name>
          <visibility>1</visibility>
          <description>Waypoint</description>
          <styleUrl>#waypointStyle</styleUrl>
          <ExtendedData xmlns:mis="www.dji.com">
            <mis:useWaylineAltitude>false</mis:useWaylineAltitude>
            <mis:heading>$heading</mis:heading>
            <mis:turnMode>$turnmode</mis:turnMode>
            <mis:gimbalPitch>$gimbal</mis:gimbalPitch>
            <mis:useWaylineSpeed>false</mis:useWaylineSpeed>
            <mis:speed>$speed</mis:speed>
            <mis:useWaylineHeadingMode>true</mis:useWaylineHeadingMode>
            <mis:useWaylinePointType>true</mis:useWaylinePointType>
            <mis:pointType>LineStop</mis:pointType>
            <mis:cornerRadius>0.2</mis:cornerRadius>""")

    waypoint_end = Template("""
          </ExtendedData>
          <Point>
            <altitudeMode>relativeToGround</altitudeMode>
            <coordinates>$lon,$lat,$height</coordinates>
          </Point>
        </Placemark>""")
    hover_template = Template("""
            <mis:actions param="$length" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">Hovering</mis:actions>""")
    shoot_template = Template("""
            <mis:actions param="0" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">ShootPhoto</mis:actions>""")

    gimbal_template = Template("""
            <mis:actions param="$gimbal_angle" accuracy="1" cameraIndex="0" payloadType="0" payloadIndex="0">GimbalPitch</mis:actions>""")
    aircraftyaw_template = Template("""
            <mis:actions param="$aircraftyaw" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">AircraftYaw</mis:actions>""")
    record_template = Template("""
            <mis:actions param="0" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">StartRecording</mis:actions>""")
    stoprecord_template = Template("""
            <mis:actions param="0" accuracy="0" cameraIndex="0" payloadType="0" payloadIndex="0">StopRecording</mis:actions>""")

    all_coordinates_template = Template("$lon,$lat,$height")
    xml_end = Template("""    </Folder>
      <Placemark>
        <name>Wayline</name>
        <description>Wayline</description>
        <visibility>1</visibility>
        <ExtendedData xmlns:mis="www.dji.com">
          <mis:altitude>50.0</mis:altitude>
          <mis:autoFlightSpeed>5.0</mis:autoFlightSpeed>
          <mis:actionOnFinish>$ON_FINISH</mis:actionOnFinish>
          <mis:headingMode>UsePointSetting</mis:headingMode>
          <mis:gimbalPitchMode>UsePointSetting</mis:gimbalPitchMode>
          <mis:powerSaveMode>false</mis:powerSaveMode>
          <mis:waypointType>LineStop</mis:waypointType>
          <mis:droneInfo>
            <mis:droneType>COMMON</mis:droneType>
            <mis:advanceSettings>false</mis:advanceSettings>
            <mis:droneCameras/>
            <mis:droneHeight>
              <mis:useAbsolute>false</mis:useAbsolute>
              <mis:hasTakeoffHeight>false</mis:hasTakeoffHeight>
              <mis:takeoffHeight>0.0</mis:takeoffHeight>
            </mis:droneHeight>
          </mis:droneInfo>
        </ExtendedData>
        <styleUrl>#waylineGreenPoly</styleUrl>
        <LineString>
          <tessellate>1</tessellate>
          <altitudeMode>relativeToGround</altitudeMode>
          <coordinates>$all_coordinates</coordinates>
        </LineString>
      </Placemark>
    </Document>
  </kml>""")

    with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        writer = csv.writer(temp_file)
        writer.writerows(coordinates)

    with open(temp_file.name, newline='', mode='r') as csvfile:
        if csv.Sniffer().has_header(csvfile.read(1024)):
            CSV_HEADER = True
        csvfile.seek(0)
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csv_lines = csv.reader(csvfile, dialect)
        if CSV_HEADER:
            next(csv_lines, None)  # skip the headers

        for row in csv_lines:
            if row:
                print(row)
                name = row[0]
                lon = row[1]
                lat = row[2]
                if lon[0] == '_':
                    lon = lon[1:]
                if lat[0] == '_':
                    lon = lat[1:]
                height = row[3]
                heading = row[4]
                gimbal = row[5]
                speed = row[6]
                turnmode = row[7]
                actions_sequence = row[8]

                if (float(speed) > 15) or (float(speed) <= 0):
                    raise ValueError('speed should be >0 or <=15 m/s for {}'.format(name))
                if '.' not in speed:
                    speed = speed+'.0'

                if '.' not in gimbal:
                    gimbal = gimbal+'.0'

                if turnmode == 'AUTO':
                    turnmode = 'Auto'
                elif turnmode == 'C':
                    turnmode = 'Clockwise'
                elif turnmode == 'CC':
                    turnmode = 'Counterclockwise'
                else:
                    raise ValueError('turnmode shoud be AUTO C or CC for {}'.format(name))

                XML_string += waypoint_start.substitute(
                    turnmode=turnmode, waypoint_number=waypoint_number, speed=speed, heading=heading, gimbal=gimbal)

                # Actions decoding
                if actions_sequence:
                    action_list = actions_sequence.split('.')
                    for action in action_list:
                        if action == 'SHOOT':
                            XML_string += shoot_template.substitute()
                        elif action == 'REC':
                            XML_string += record_template.substitute()
                        elif action == 'STOPREC':
                            XML_string += stoprecord_template.substitute()
                        # Gimbal orientation
                        elif action[0] == 'G':
                            XML_string += gimbal_template.substitute(
                                gimbal_angle=action[1:])
                        # Aircraft orientation
                        elif action[0] == 'A':
                            XML_string += aircraftyaw_template.substitute(
                                aircraftyaw=action[1:])
                        elif action[0] == 'H':
                            if float(action[1:]) < 500:
                                print(float(action[1:]))
                                raise ValueError('Hover length is in ms and should be >500  for {}'.format(name))
                            XML_string += hover_template.substitute(
                                length=action[1:])

                XML_string += "\n" + \
                    waypoint_end.substitute(lon=lon, lat=lat, height=height,)+"\n"

                all_coordinates += all_coordinates_template.substitute(
                    lon=lon, lat=lat, height=height)+" "
            waypoint_number += 1

    Path(temp_file.name).unlink(missing_ok=False)

    # remove last space from coordinates string
    all_coordinates = all_coordinates[:-1]
    XML_string += xml_end.substitute(all_coordinates=all_coordinates,
                                     ON_FINISH=on_finish)

    with open(path_output_file, 'w') as outpufile:
        outpufile.write(XML_string)
