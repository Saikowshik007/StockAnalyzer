docker build -t financial-news-monitor .

docker run -d --name financial-monitor --env-file .env -v ./database:/app/database -v ./logs:/app/logs -v ./backups:/app/backups financial-news-monitor