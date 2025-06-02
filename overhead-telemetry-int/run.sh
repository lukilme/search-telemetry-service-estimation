#!/bin/bash

vagrant ssh host-2 -c "cd /vagrant/code && sudo timeout 3600 ./receive.py" &

vagrant ssh host-1 -c "cd /vagrant/code && sudo timeout 3600 ./send.py 192.168.50.12" &

vagrant ssh host-2 -c "export DISPLAY=:0 && timeout 3600 vlc http://192.168.50.11/videos/worldcup2002-mp4.mpd" &

vagrant ssh host-3 -c "export DISPLAY=:0 && cd /vagrant/logs && python2.7 sinusoid.py -l http://192.168.50.11/videos/worldcup2002-mp4.mpd -s 3,2 60 5"
