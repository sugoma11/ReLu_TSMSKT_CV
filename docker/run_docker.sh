#!/usr/bin/env bash

docker run --rm -it \
    --net=host \
    --runtime=nvidia \
    -p 8080:8080 \
    --privileged \
    sauvc_autostart:latest \
    bash

echo "Done."
