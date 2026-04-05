.PHONY: help install run init-db test clean docker-up docker-down

help:
	@echo "Доступные команды:"
	@echo "  make install    - Установка зависимостей"
	@echo "  make run        - Запуск бота"
	@echo "  make init-db    - Инициализация базы данных"
	@echo "  make test       - Запуск тестов"
	@echo "  make docker-up  - Запуск PostgreSQL в Docker"
	@echo "  make docker-down- Остановка PostgreSQL"

install:
	pip install -r requirements.txt

run:
	python -m src.main

init-db:
	python scripts/init_db.py

test:
	pytest tests/ -v

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down