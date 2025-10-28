# process_data.py
import sys, csv

out = csv.writer(open("sensor_log.csv", "w", newline=""))
out.writerow(["t_ms","r","g","b","c","temperatureC","humidityPct","accelZ_mps2","aqi","tvoc_ppb","eco2_ppm"])
print("process_data.py: READY", flush=True)

try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # ignore headers or chatter
        if line.startswith("HEADER,"):
            continue
        if not line.startswith("DATA,"):
            continue

        fields = line.split(",", maxsplit=12)[1:]  # everything after "DATA,"
        # Expecting 11 fields based on our CSV
        if len(fields) != 11:
            continue

        try:
            t_ms      = int(fields[0])
            r         = int(fields[1])
            g         = int(fields[2])
            b         = int(fields[3])
            c         = int(fields[4])
            temperature    = float(fields[5])
            humidity   = float(fields[6])
            accel_z   = float(fields[7])
            aqi       = int(fields[8])
            tvoc  = int(fields[9])
            eco2  = int(fields[10])
        except ValueError:
            continue

        # Write to CSV (or do your own processing instead)
        out.writerow([t_ms,r,g,b,c,temperature,humidity,accel_z,aqi,tvoc,eco2])

        # Example real-time print
        print(f"T={t_ms}ms  Temp={temperature:.2f}C  RH={humidity:.1f}%  eCO2={eco2}ppm  TVOC={tvoc}ppb", flush=True)

except KeyboardInterrupt:
    pass

print("process_data.py: EXIT", flush=True)
