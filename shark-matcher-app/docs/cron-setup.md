# Cron Job Setup for Shark Matcher

This document explains how to set up automatic database snapshots using cron
jobs. The schedule and generated wrapper are defined in `scripts/setup_cron.sh`.

While the app server runs, it already writes the same complete trusted backup
each Sunday between midnight and 1 AM. The cron job takes that backup whether or
not the server is running, so with both in place each week produces two copies.

## Quick Setup with Automatic Setup Script
Run the setup script from the checkout or invoke it by its relative location:
```bash
./scripts/setup_cron.sh
```

This script will:
- Automatically detect your project directory
- Create or refresh a self-locating wrapper script
- Prepare the configured cron job with the resolved wrapper location
- Preserve unrelated entries and replace only its own snapshot block
- Record the Python interpreter, and `SHARK_DATA_ROOT` when it is set, in that
  block, because cron runs with a minimal environment
- Ask for confirmation before installing the proposed crontab

The wrapper and log are prepared even if you decline installation.

- The script reads the current crontab, displays the complete proposed result,
  and preserves entries outside its managed snapshot block.
- If it cannot safely read the current crontab or finds malformed snapshot
  markers, it exits without installing anything.
- Review the proposed content before accepting it.

## Managing the Cron Job

### View Current Cron Jobs
```bash
crontab -l
```

### Edit or Remove the Snapshot Job
```bash
crontab -e
```

To remove only this snapshot job, delete its generated snapshot block while
leaving unrelated jobs intact. Do not use `crontab -r` for this purpose: it
removes every job in your crontab.

## Monitoring Logs

The wrapper logs each run to `data-out/logs/cron_output.txt` inside the
checkout. The snapshot command itself also appends to
`data-out/logs/shark_identification.log` under the data root, and the backup
files are written beneath the data root's `data-out/` directory.

### View Recent Log Entries
```bash
# View the last 10 log entries
tail -n 10 data-out/logs/cron_output.txt

# Follow the log file in real-time
tail -f data-out/logs/cron_output.txt

# View all log entries
cat data-out/logs/cron_output.txt
```

### Log Format
The generated wrapper writes the system `date` output followed by the snapshot
start and either a success or failure message. See the wrapper template in
`scripts/setup_cron.sh` for the current message format. The first setup may also write
a log-initialization line.

## Testing

### Test the Script Manually
From the checkout root with the app environment active, run the command the
wrapper runs:
```bash
python3 -m scripts.shark_identification -s
```

### Test Without Installing a Cron Job
```bash
# Generate the wrapper and review the proposed cron entry.
./scripts/setup_cron.sh
# Answer "n" when prompted so the crontab is not changed.

# Run the generated wrapper directly when you are ready to test a snapshot.
./scripts/cron_wrapper.sh
tail -f data-out/logs/cron_output.txt
```

## Troubleshooting

### Check Cron Logs
```bash
# On macOS
log show --predicate 'process == "cron"' --last 1d

# On Linux
grep CRON /var/log/syslog
```

### Common Issues

1. **Permission Denied**: Run the snapshot entry point with Python, or check
   permissions on the checkout and backup destination.
   ```bash
   python3 -m scripts.shark_identification -s
   ```

2. **Path Not Found**: Confirm the setup script and application entry point are
   in the same checkout
   ```bash
   ls scripts/setup_cron.sh scripts/shark_identification.py
   ```

3. **Python Not Found**: The setup script records the interpreter that
   `python3` resolves to when you run it (printed as "Using Python
   interpreter"). Run it with the app's virtual environment active, or name the
   interpreter explicitly from the checkout root:
   ```bash
   SHARK_PYTHON="$PWD/.venv/bin/python3" ./scripts/setup_cron.sh
   ```

## Multi-Machine Setup

Run the automatic setup script once in each checkout. The generated wrapper
resolves its own location at runtime, so no machine-specific project path needs
to be copied into the script or documentation.

## Security Notes

- Cron jobs run with minimal environment variables
- Make sure your script paths are secure
- Database snapshots are trusted server-side backups and may contain sensitive
  data. Restrict access to their destination; do not expose them to clients.
- Consider running with a specific user account if needed
- Test thoroughly before deploying to production
