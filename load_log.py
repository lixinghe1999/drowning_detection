# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:56:36 2020

@author: HeLix
"""
import struct
from brping import PingParser
from dataclasses import dataclass
from typing import IO, Any
from sonar_display import get_data,show_sonar
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class PingViewerBuildInfo:
    hash_commit: str = ''
    date: str = ''
    tag: str = ''
    os_name: str = ''
    os_version: str = ''

    def __str__(self):
        return 


@dataclass
class Sensor:
    family: int = 0
    type_sensor: int = 0

    def __str__(self):
        return 


@dataclass
class Header:
    string: str = ''
    version: int = 0
    ping_viewer_build_info = PingViewerBuildInfo()
    sensor = Sensor()

    def __str__(self):
        return f"""Header:
    String: {self.string}
    Version: {self.version}
    PingViewerBuildInfo:
        hash: {self.ping_viewer_build_info.hash_commit}
        date: {self.ping_viewer_build_info.date}
        tag: {self.ping_viewer_build_info.tag}
        os:
            name: {self.ping_viewer_build_info.os_name}
            version: {self.ping_viewer_build_info.os_version}
    Sensor:
        Family: {self.sensor.family}
        Type: {self.sensor.type_sensor}
    """


class Log:
    def __init__(self, filename: str):
        self.filename = filename
        self.header = Header()
        self.messages = []

    @staticmethod
    def unpack_int(file: IO[Any]):
        version_format = '>1i'
        data = file.read(struct.calcsize(version_format))
        return struct.Struct(version_format).unpack_from(data)[0]

    @staticmethod
    def unpack_array(file: IO[Any]):
        array_size = Log.unpack_int(file)
        return file.read(array_size)

    @staticmethod
    def unpack_string(file: IO[Any]):
        return Log.unpack_array(file).decode('UTF-8')

    @staticmethod
    def unpack_message(file: IO[Any]):
        timestamp = Log.unpack_string(file)
        message = Log.unpack_array(file)
        return (timestamp, message)

    def unpack_header(self, file: IO[Any]):
        self.header.string = self.unpack_string(file)
        self.header.version = self.unpack_int(file)

        self.header.ping_viewer_build_info.hash_commit = self.unpack_string(
            file)
        self.header.ping_viewer_build_info.date = self.unpack_string(file)
        self.header.ping_viewer_build_info.tag = self.unpack_string(file)
        self.header.ping_viewer_build_info.os_name = self.unpack_string(file)
        self.header.ping_viewer_build_info.os_version = self.unpack_string(
            file)

        self.header.sensor.family = self.unpack_int(file)
        self.header.sensor.type_sensor = self.unpack_int(file)

    def process(self):
        with open(self.filename, "rb") as file:
            self.unpack_header(file)
            while True:
                try:
                    self.messages.append(log.unpack_message(file))
                except Exception as _:
                    break

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Control the sonar')
    parser.add_argument('--loc', action="store", required=False, type=str, default='sensor_log/log_3.bin',
                        help="The location of binary file")
    args = parser.parse_args()
    bin_loc=args.loc
    # Open log and process it

    log = Log(bin_loc)
    log.process()
    #print(log.header)
    fig=plt.figure(1)
    ping_parser = PingParser()
    i=0
    for (timestamp, message) in log.messages:
        i+=1
        # Parse each byte of the message
        for byte in message:
            # Check if the parser has a new message
            if ping_parser.parse_byte(byte) is PingParser.NEW_MESSAGE:
                # Get decoded message
                decoded_message = ping_parser.rx_msg
                # Filter for the desired ID
                # 1300 for Ping1D profile message and 2300 for Ping360
                if decoded_message.message_id in [1300, 2300]:
                    if i==1:
                        d=len(data)
                        sonar_img = np.zeros((d, 400))
                    data,angle=get_data(decoded_message)
                    sonar_img[:,angle]=data
        if i >=2000:
            show_sonar(sonar_img, 2, fig)
            plt.show()
            break
