#!/bin/zsh

DIR_ACTUAL=$(pwd)

MOJO_DIR="/home/kilmer/Development/lab/mojo-tests/hello-world"

(cd "$MOJO_DIR" && magic shell)

cd "$DIR_ACTUAL"