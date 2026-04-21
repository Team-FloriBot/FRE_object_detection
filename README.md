generate synthetic:
docker compose --profile gen up --build

train:
docker compose --profile train up --build

detect:
docker compose --profile detect up --build