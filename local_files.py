#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys
import os
import pathlib
import argparse

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serve files in directory (allowing Cross-Origin Requests)')
    parser.add_argument('-p', '--port', help='default: 8081', default=8081, type=int)
    parser.add_argument('--directory',
                        default=pathlib.Path.cwd(),
                        help='directory to serve (default: current working directory)',
                        type=pathlib.Path)
    args = parser.parse_args()

    port = args.port
    directory = args.directory

    if not directory.exists():
        sys.stderr.write(f"Error: Directory {directory} does not exist\n")
        exit(1)

    if not directory.is_dir():
        sys.stderr.write(f"Error: {directory} is not a directory\n")
        exit(1)

    # Change the working directory to the specified directory
    os.chdir(directory)

    # Start the server
    test(CORSRequestHandler, HTTPServer, port=port)
