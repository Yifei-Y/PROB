```
cd docker/
# Build:
docker build --build-arg USER_ID=$UID -t prob:v0 .
# Launch (require GPUs):
docker run -i -d \
  --shm-size=64gb --gpus all \
  --network host \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /nas_data/yyf/workspace/PROB:/home/appuser/PROB \
  -v /nas_data/yyf/dataset:/home/appuser/Datasets \
  -v /etc/localtime:/etc/localtime \
  -w /home/appuser \
  --name=PROB prob:v0
# Enter container
docker exec -it PROB /bin/bash