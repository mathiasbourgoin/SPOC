#!/bin/bash
docker run -e DISPLAY=unix:0.0 -v=/tmp/.X11-unix:/tmp/.X11-unix:rw --privileged -i -t spoc_docker 
