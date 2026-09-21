cd /root/paper5
source /root/nselib_env/bin/activate
nohup python3.11 scripts/backtest_bullput_sweep.py > results/sweep_bullput.log 2>&1 &
echo $! > /tmp/sweep.pid
echo "Sweep running in background, PID $(cat /tmp/sweep.pid)"

