 build:
	 sudo docker-compose build

run:
	sudo docker-compose up -d
	sudo docker-compose exec lispy bash

down:
	docker-compose -f docker-compose.yml down