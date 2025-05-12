{ # try
	source .venv/bin/activate
} || { # catch
    python3.10 --version
    python3.10 -m pip --version
  	python3.10 -m pip install uv
	uv venv
	source .venv/bin/activate
}