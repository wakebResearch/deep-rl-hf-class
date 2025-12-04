!/usr/bin/bash

uv sync
uv run rl_zoo3 enjoy  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --no-render  --n-timesteps 5000  --folder logs/
