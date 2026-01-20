# Test the full pipeline
python -m src.cli.record  # Record something simple
python -m src.cli.compile --session <session_id> --name test --llm
python -m src.cli.replay --recipe test.json --params '{"query": "pizza"}'