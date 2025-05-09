docker build -t financial-news-monitor .
docker stop financial-monitor
docker rm financial-monitor
docker run -d --name financial-monitor \
  --env-file ../env_files/.env \
  -v ./database:/app/database \
  -v ./logs:/app/logs \
  -v ./backups:/app/backups \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  financial-news-monitor