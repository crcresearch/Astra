# listener.py
import serial, subprocess, sys, signal

PORT = "/dev/ttyACM0"   # Linux/Jetson example; Windows like "COM3"; macOS like "/dev/tty.usbmodemXXXX"
BAUD = 115200

proc = None  # subprocess for process_data.py

def start_processing():
    global proc
    if proc is None or proc.poll() is not None:
        # Launch process_data.py and open a text pipe for input
        proc = subprocess.Popen(
            [sys.executable, "process_data.py"],
            stdin=subprocess.PIPE, text=True, bufsize=1
        )
        print("Started process_data.py")

def stop_processing():
    global proc
    if proc and proc.poll() is None:
        print("Stopping process_data.py ...")
        try:
            # Close its stdin so it can exit gracefully
            proc.stdin.close()
        except Exception:
            pass
        # Politely ask it to stop
        proc.send_signal(signal.SIGINT)

def main():
    global proc
    print(f"Opening serial {PORT} @ {BAUD} ...")
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print(f"Listening on {PORT} ...")
        while True:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()

            if line == "START":
                print("[SERIAL] START")
                start_processing()
            elif line == "STOP":
                print("[SERIAL] STOP")
                stop_processing()
                proc = None
            elif line.startswith("DATA,"):
                # Forward DATA lines only if the processor is running
                if proc and proc.poll() is None:
                    try:
                        proc.stdin.write(line + "\n")
                        proc.stdin.flush()
                    except BrokenPipeError:
                        # Child exited; drop data until next START
                        proc = None
                # (Optionally print a dot or echo for debugging)
            else:
                # Unknown line; ignore or log
                pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_processing()
        print("\nListener exiting.")
