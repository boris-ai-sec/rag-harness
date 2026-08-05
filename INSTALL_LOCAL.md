# LAB-RH-01 local bootstrap instructions

Approved target: Ubuntu under WSL 2, project path `~/ai-projects/rag-harness`.

## 1. Confirm the target directory

```bash
cd ~/ai-projects/rag-harness
pwd
find . -mindepth 1 -maxdepth 1 -print
```

The final command should currently produce no output.

## 2. Keep Miniconda out of the Harness environment

If the prompt shows a Conda environment, deactivate it before continuing:

```bash
conda deactivate
printf 'CONDA_PREFIX=%s\n' "${CONDA_PREFIX:-}"
```

An empty `CONDA_PREFIX` is the expected result. Do not install Harness packages into Conda `base`.

## 3. Install the approved CPython 3.12 provider

The approved path is a user-local installation of Astral `uv`. It installs a managed CPython build without changing Ubuntu's system Python or the Miniconda installation.

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
sh /tmp/uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.12
uv python find 3.12
```

Record the complete output of `uv --version` and `uv python find 3.12` in the laboratory observations.

## 4. Extract this bootstrap package

Extract the package contents directly into the empty target directory. From a typical Windows Downloads directory, replace `<windows-user>` if needed:

```bash
cd ~/ai-projects/rag-harness
unzip /mnt/c/Users/<windows-user>/Downloads/rag-harness-bootstrap-v0.1.zip -d .
find . -maxdepth 3 -type f -print | sort
```

If the package is already stored elsewhere in WSL, use its actual path in the `unzip` command.

## 5. Create the isolated Harness environment

```bash
cd ~/ai-projects/rag-harness
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
python -c 'import sys; print(sys.executable); print(sys.prefix)'
bash scripts/check_environment.sh
```

Acceptance criteria:

- `python --version` reports Python 3.12.x;
- the executable is under `~/ai-projects/rag-harness/.venv/`;
- the environment check finishes with `PASS`;
- `CONDA_PREFIX` is empty.

## 6. Validate and start Qdrant

```bash
cd ~/ai-projects/rag-harness
docker compose config
docker compose pull qdrant
docker compose up -d qdrant
docker compose ps
bash scripts/check_qdrant.sh
docker volume inspect rag_harness_qdrant_data
```

Qdrant REST and gRPC are exposed only on the local machine at `127.0.0.1:6333` and `127.0.0.1:6334`. The dashboard is available locally at `http://127.0.0.1:6333/dashboard`.

## 7. Persistence check

```bash
docker compose restart qdrant
bash scripts/check_qdrant.sh
docker compose ps
```

To stop Qdrant without deleting data:

```bash
docker compose stop qdrant
```

Do not use `docker compose down -v`; `-v` deletes the named data volume.

## 8. Return the laboratory observations

Provide Harness Build with:

- `uv --version`;
- `python --version` and the Python executable path;
- `docker compose config --images`;
- `docker compose ps`;
- Qdrant readiness-check output;
- named-volume name;
- any deviations, warnings, or failed commands.

Do not include secrets, complete environment-variable dumps, or unrelated host information.
