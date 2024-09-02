#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: script_name.py [port] [directory_to_serve]")
        sys.exit(1)

    port = int(sys.argv[1])
    directory = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

    # Change the working directory to the specified directory
    os.chdir(directory)

    # Start the server
    test(CORSRequestHandler, HTTPServer, port=port)
