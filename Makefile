activate:
	@.venv\scripts\activate.bat

freeze:
	@uv pip freeze > requirements.txt

sync:
	@uv sync

lock:
	@uv lock

push:
	@git push

add:
	@git add .

commit:
	@git commit -m "update"	

status:
	@git status

git:
	@git add .
	@git commit -m "update"