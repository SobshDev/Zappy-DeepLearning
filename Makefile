# Zappy-DeepLearning — training & verification entry points.
# `make train` launches the detached overnight curriculum chain (see
# tools/overnight_train.sh): 12x12/3p -> 16x16/4p -> 20x24/6p x2, each segment
# warm-started from the previous checkpoint. ~7.5 h on the 4090.

# bash, not dash: the stop/status targets need `kill -- -PGID` (negative pid =
# whole process group), which dash's kill builtin cannot parse.
SHELL   := /bin/bash

PY      := .venv/bin/python
CPU_ENV := JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
PIDFILE := runs/overnight.pid
LOG     := runs/overnight-$(shell date +%Y%m%d-%H%M%S).log
SR_PID  := runs/speedrun.pid
SR_LOG  := runs/speedrun-$(shell date +%Y%m%d-%H%M%S).log

.PHONY: help train train-dry watch status stop test validate gate smoke clean-smoke \
	speedrun speedrun-watch speedrun-status speedrun-stop

help:
	@echo "make train       - launch the overnight curriculum chain (detached, survives logout)"
	@echo "make watch       - tail the running overnight log"
	@echo "make status      - is it alive? last log lines + GPU usage"
	@echo "make stop        - stop the overnight chain (completed segments are kept;"
	@echo "                   'make train' later resumes after them)"
	@echo "make train-dry   - 3-min dry run of the same chain (mechanics test)"
	@echo "make speedrun    - train-until-plateau toward the time-to-L8 limit (detached)"
	@echo "make speedrun-watch / speedrun-status / speedrun-stop"
	@echo "make smoke       - 2-min single-stage GPU training smoke"
	@echo "make test        - full pytest suite on CPU"
	@echo "make validate    - oracle + obs-contract validation vs the live reference server"
	@echo "make gate        - deploy gate: frozen policy vs the live reference server"

train:
	@if [ -f $(PIDFILE) ] && kill -0 -- -$$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "already running (pgid $$(cat $(PIDFILE))) — make watch / make stop"; exit 1; fi
	@mkdir -p runs
	@setsid nohup sh tools/overnight_train.sh > $(LOG) 2>&1 & \
		echo $$! > $(PIDFILE)
	@ln -sf $(notdir $(LOG)) runs/overnight-latest.log
	@sleep 2
	@echo "overnight chain launched (pgid $$(cat $(PIDFILE))) -> $(LOG)"
	@echo "follow with: make watch    stop with: make stop"

train-dry:
	DRY=1 sh tools/overnight_train.sh

speedrun:
	@if [ -f $(SR_PID) ] && kill -0 -- -$$(cat $(SR_PID)) 2>/dev/null; then \
		echo "already running (pgid $$(cat $(SR_PID))) — make speedrun-watch / speedrun-stop"; exit 1; fi
	@mkdir -p runs
	@setsid nohup $(PY) tools/speedrun_train.py > $(SR_LOG) 2>&1 & \
		echo $$! > $(SR_PID)
	@ln -sf $(notdir $(SR_LOG)) runs/speedrun-latest.log
	@sleep 2
	@echo "speedrun launched (pgid $$(cat $(SR_PID))) -> $(SR_LOG)"
	@echo "follow with: make speedrun-watch    stop with: make speedrun-stop"

speedrun-watch:
	tail -f runs/speedrun-latest.log

speedrun-status:
	@if [ -f $(SR_PID) ] && kill -0 -- -$$(cat $(SR_PID)) 2>/dev/null; \
		then echo "RUNNING (pgid $$(cat $(SR_PID)))"; \
		else echo "NOT RUNNING"; fi
	@tail -n 8 runs/speedrun-latest.log 2>/dev/null || true
	@nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true

speedrun-stop:
	@if [ -f $(SR_PID) ]; then \
		kill -TERM -- -$$(cat $(SR_PID)) 2>/dev/null && echo "stopped pgid $$(cat $(SR_PID))" \
			|| echo "was not running"; \
		rm -f $(SR_PID); \
	else echo "no pidfile"; fi

watch:
	tail -f runs/overnight-latest.log

status:
	@if [ -f $(PIDFILE) ] && kill -0 -- -$$(cat $(PIDFILE)) 2>/dev/null; \
		then echo "RUNNING (pgid $$(cat $(PIDFILE)))"; \
		else echo "NOT RUNNING"; fi
	@tail -n 8 runs/overnight-latest.log 2>/dev/null || true
	@nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true

stop:
	@if [ -f $(PIDFILE) ]; then \
		kill -TERM -- -$$(cat $(PIDFILE)) 2>/dev/null && echo "stopped pgid $$(cat $(PIDFILE))" \
			|| echo "was not running"; \
		rm -f $(PIDFILE); \
	else echo "no pidfile"; fi

smoke:
	XLA_PYTHON_CLIENT_PREALLOCATE=false $(PY) -m zappy_rl.train \
		--run-name smoke12x12-3p --width 12 --height 12 --n-agents 3 \
		--max-episode-ticks 8192 --eval-max-ticks 8192 --ent-coef-token 0.01 \
		--total-env-steps 2000000 --eval-envs 64 --wandb disabled \
		--init-actor runs/ritual8x8-v1/params.msgpack

test:
	$(CPU_ENV) $(PY) -m pytest -q

validate:
	$(CPU_ENV) $(PY) tools/validate_against_server.py
	$(CPU_ENV) $(PY) tools/validate_obs_contract.py

gate:
	$(PY) tools/run_deploy_gate.py

clean-smoke:
	rm -rf runs/smoke* runs/*-dry
