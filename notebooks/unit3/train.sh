!/usr/bin/bash

uv sync
uv run  rl_zoo3 train --algo dqn  --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml
