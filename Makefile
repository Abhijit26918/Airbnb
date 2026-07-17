.PHONY: setup data clean-data features train eval audit app all test lint attribution-check

UV := uv

setup:
	$(UV) sync --all-extras
	git config core.hooksPath .githooks
	chmod +x .githooks/commit-msg .githooks/pre-commit

data:
	$(UV) run pricelens data download

clean-data:
	$(UV) run pricelens data clean

features:
	$(UV) run pricelens features build

train:
	$(UV) run pricelens model train

eval:
	$(UV) run pricelens model eval

audit:
	$(UV) run pricelens validate audit

app:
	docker compose up

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

attribution-check:
	git log --format='%h | %an <%ae> | %s%n%b' -20 | grep -iE 'claude|anthropic|co-authored|generated with' && exit 1 || echo OK

all: setup data clean-data features train eval audit
