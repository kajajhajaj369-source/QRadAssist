"""Autonomous monitor for QRadAssist Phase 9 full GPU training.

Watches the running `train_full.py` process and `experiments/full_gpu/training.log`.
Sends native Windows desktop notifications (Toast with Balloon Tip fallback):
1. When training successfully finishes (including epoch count, AUROC, checkpoint paths).
2. When training fails, crashes, or exits unexpectedly.
3. When training appears stalled (no log activity for a configurable threshold, e.g. 15 mins).

Usage:
    python monitor_training.py [--stall-minutes 15] [--poll-interval 20] [--test-notify]
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
FULL_DIR = ROOT / "experiments" / "full_gpu"
LOG_FILE = FULL_DIR / "training.log"
BEST_CKPT = FULL_DIR / "best_full_model.pt"
MONITOR_LOG = FULL_DIR / "monitor.log"


def log_monitor(msg: str) -> None:
    """Write timestamped log to console and experiments/full_gpu/monitor.log."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{now}] [Monitor] {msg}"
    print(formatted, flush=True)
    try:
        FULL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MONITOR_LOG, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def send_windows_notification(title: str, message: str) -> bool:
    """Send native Windows desktop notification via PowerShell WinRT Toast,
    with a System.Windows.Forms balloon tip fallback."""
    log_monitor(f"SENDING NOTIFICATION: {title} | {message}")

    ps_script = f"""
$Title = @'
{title}
'@
$Message = @'
{message}
'@

function Show-Toast {{
    param($Title, $Message)
    try {{
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
        $template = @"
<toast>
    <visual>
        <binding template="ToastGeneric">
            <text>$Title</text>
            <text>$Message</text>
        </binding>
    </visual>
</toast>
"@
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml($template)
        $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
        $appId = "{{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}}\\WindowsPowerShell\\v1.0\\powershell.exe"
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)
        Write-Host "TOAST_OK"
        return $true
    }} catch {{
        Write-Host "TOAST_ERR: $_"
        return $false
    }}
}}

function Show-Balloon {{
    param($Title, $Message)
    try {{
        Add-Type -AssemblyName System.Windows.Forms
        $notify = New-Object System.Windows.Forms.NotifyIcon
        $notify.Icon = [System.Drawing.SystemIcons]::Information
        $notify.BalloonTipTitle = $Title
        $notify.BalloonTipText = $Message
        $notify.Visible = $True
        $notify.ShowBalloonTip(10000)
        Start-Sleep -Milliseconds 500
        $notify.Dispose()
        Write-Host "BALLOON_OK"
        return $true
    }} catch {{
        Write-Host "BALLOON_ERR: $_"
        return $false
    }}
}}

$ok = Show-Toast -Title $Title -Message $Message
if (-not $ok) {{
    Show-Balloon -Title $Title -Message $Message
}}
"""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = proc.stdout.strip()
        log_monitor(f"Notification dispatch result: {output}")
        return "TOAST_OK" in output or "BALLOON_OK" in output
    except Exception as exc:
        log_monitor(f"Failed to dispatch notification: {exc}")
        return False


def find_training_processes() -> list[psutil.Process]:
    """Find running train_full.py processes."""
    matched = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = p.info.get("cmdline") or []
            cmd_str = " ".join(cmd).lower()
            if "train_full.py" in cmd_str and p.pid != os.getpid():
                matched.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matched


def parse_latest_log_state() -> dict:
    """Parse experiments/full_gpu/training.log for current status."""
    res = {
        "exists": False,
        "mtime": 0.0,
        "last_line": "",
        "latest_epoch": 0,
        "latest_batch": 0,
        "total_batches": 21505,
        "completed": False,
        "mean_auc": None,
        "has_error": False,
    }
    if not LOG_FILE.is_file():
        return res

    res["exists"] = True
    try:
        res["mtime"] = LOG_FILE.stat().st_mtime
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if not lines:
            return res

        res["last_line"] = lines[-1]

        # Scan for epoch/batch lines: [Epoch 1/5]  batch 1700/21505 ...
        batch_re = re.compile(r"\[Epoch (\d+)/(\d+)\]\s+batch (\d+)/(\d+)")
        mean_auc_re = re.compile(r"Mean AUROC.*:\s+([\d\.]+)")

        for line in reversed(lines):
            if "Traceback (most recent call last)" in line or "Error:" in line:
                res["has_error"] = True
            if "Training completed" in line or "=== Final Validation AUROC ===" in line:
                res["completed"] = True
            m_auc = mean_auc_re.search(line)
            if m_auc and res["mean_auc"] is None:
                res["mean_auc"] = float(m_auc.group(1))
            m_batch = batch_re.search(line)
            if m_batch and res["latest_epoch"] == 0:
                res["latest_epoch"] = int(m_batch.group(1))
                res["latest_batch"] = int(m_batch.group(3))
                res["total_batches"] = int(m_batch.group(4))

    except Exception as exc:
        log_monitor(f"Error parsing log file: {exc}")

    return res


def monitor_loop(stall_minutes: float = 15.0, poll_seconds: float = 20.0) -> None:
    """Main monitoring loop."""
    stall_timeout = stall_minutes * 60.0
    log_monitor(f"Starting QRadAssist Training Monitor (Stall threshold: {stall_minutes}m, Poll: {poll_seconds}s)")
    log_monitor(f"Watching Log: {LOG_FILE}")
    log_monitor(f"Watching Output Dir: {FULL_DIR}")

    stall_alerted = False
    last_progress_time = time.time()
    last_seen_state_str = ""

    # Initial process check
    procs = find_training_processes()
    if procs:
        pids = [p.pid for p in procs]
        log_monitor(f"Detected active training process(es): PIDs {pids}")
    else:
        log_monitor("WARNING: No running train_full.py process detected at startup!")

    while True:
        try:
            procs = find_training_processes()
            is_running = len(procs) > 0
            state = parse_latest_log_state()

            current_state_str = (
                f"ep={state['latest_epoch']} batch={state['latest_batch']} "
                f"last='{state['last_line'][:60]}'"
            )

            # Check for forward progress in log
            if current_state_str != last_seen_state_str:
                last_seen_state_str = current_state_str
                last_progress_time = time.time()
                if stall_alerted:
                    log_monitor("Progress resumed; resetting stall alert state.")
                    stall_alerted = False
                log_monitor(f"Progress update: [Epoch {state['latest_epoch']}/5] batch {state['latest_batch']}/{state['total_batches']} | {state['last_line']}")

            # Check for Successful Completion
            if state["completed"]:
                log_monitor("SUCCESS DETECTED: Training completed successfully!")
                mean_auc_str = f"{state['mean_auc']:.4f}" if state["mean_auc"] is not None else "Recorded in log"
                title = "QRadAssist: Training Complete! [SUCCESS]"
                msg = (
                    f"5/5 epochs completed successfully.\n"
                    f"Mean AUROC: {mean_auc_str}\n"
                    f"Model: {BEST_CKPT.name}\n"
                    f"Directory: {FULL_DIR}"
                )
                send_windows_notification(title, msg)
                log_monitor("Completion notification sent. Exiting monitor.")
                return

            # Check if process exited
            if not is_running:
                log_monitor("Training process is NO LONGER RUNNING.")
                # Give a brief moment for log flushes
                time.sleep(2)
                final_state = parse_latest_log_state()
                if final_state["completed"]:
                    log_monitor("Final check confirms clean completion.")
                    title = "QRadAssist: Training Complete! [SUCCESS]"
                    msg = (
                        f"Training completed successfully.\n"
                        f"Model: {BEST_CKPT.name}\n"
                        f"Directory: {FULL_DIR}"
                    )
                    send_windows_notification(title, msg)
                else:
                    log_monitor("ALERT: Training process exited before completion marker!")
                    title = "QRadAssist: Training Process Stopped! [ALERT]"
                    msg = (
                        f"Process exited prematurely at Epoch {final_state['latest_epoch']}, "
                        f"Batch {final_state['latest_batch']}.\n"
                        f"Last log: {final_state['last_line'][:100]}\n"
                        f"See: {LOG_FILE}"
                    )
                    send_windows_notification(title, msg)
                return

            # Process is running - check for possible stall
            time_since_progress = time.time() - last_progress_time
            if time_since_progress > stall_timeout:
                if not stall_alerted:
                    stall_mins_actual = int(time_since_progress / 60)
                    log_monitor(f"STALL WARNING: No log progress for {stall_mins_actual} minutes! PIDs: {[p.pid for p in procs]}")
                    title = "QRadAssist: Possible Training Stall [WARNING]"
                    msg = (
                        f"No new progress for {stall_mins_actual}m.\n"
                        f"Process is alive (PID: {procs[0].pid}).\n"
                        f"Last progress: Epoch {state['latest_epoch']} Batch {state['latest_batch']}.\n"
                        f"Check: {LOG_FILE}"
                    )
                    send_windows_notification(title, msg)
                    stall_alerted = True  # Avoid repeating alert every tick
            else:
                # Log heartbeat occasionally
                pass

        except Exception as exc:
            log_monitor(f"Exception in monitor loop: {exc}")

        time.sleep(poll_seconds)


def main() -> None:
    p = argparse.ArgumentParser(description="QRadAssist Training Notification & Stall Monitor")
    p.add_argument("--stall-minutes", type=float, default=15.0, help="Minutes without log progress before alerting (default: 15)")
    p.add_argument("--poll-interval", type=float, default=20.0, help="Seconds between poll checks (default: 20)")
    p.add_argument("--test-notify", action="store_true", help="Send a test notification immediately and exit")
    args = p.parse_args()

    if args.test_notify:
        print("[Monitor] Testing Windows desktop notification...")
        ok = send_windows_notification(
            "QRadAssist Test Notification",
            "This is a test notification from QRadAssist Monitor. Notifications are working properly!"
        )
        if ok:
            print("[Monitor] Test notification successfully dispatched!")
            sys.exit(0)
        else:
            print("[Monitor] Notification dispatch reported failure.")
            sys.exit(1)

    monitor_loop(stall_minutes=args.stall_minutes, poll_seconds=args.poll_interval)


if __name__ == "__main__":
    main()
