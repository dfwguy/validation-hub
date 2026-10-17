"""Synthetic all-gather regression fixture for native payload/placement parsing.

Rounded zero-payload rows remain zero in parsed evidence. The separate scorer
rejects these points; it must not rewrite their message sizes or column order.
All numbers and cluster-scale hints below are constructed test values.
"""
from backend.parser import parse_log


def test_rank_rounded_zero_rows_preserve_actual_size_and_column_placement():
 report=parse_log([
  'Collective test starting: all_gather_perf_mpi\n',
  '# size count type redop root time algbw busbw #wrong time algbw busbw #wrong\n',
  '0 0 float none -1 0.20 0.00 0.00 0 0.10 0.00 0.00 0\n',
  '0 0 float none -1 0.12 0.00 0.00 0 0.11 0.00 0.00 0\n',
  '2147483648 8388608 float none -1 13421.8 160.00 157.50 0 13766.8 156.00 153.56 0\n',
  '17179869184 67108864 float none -1 53687.1 320.00 315.00 0 55063.7 312.00 307.12 0\n',
  '# Out of bounds values : 0 OK\n',
  '# Avg bus bandwidth : 116.6475\n',
 ],hints={'gpus':64,'nodes':8})
 points=report['metrics']
 assert len(points)==8 and [m['message_bytes'] for m in points[:4]]==[0]*4
 assert points[0]['time_us']==.20 and points[1]['time_us']==.10
 assert points[4]['message_bytes']==2147483648 and points[4]['busbw_gbps']==157.50
 assert points[5]['placement']=='in_place' and points[5]['time_us']==13766.8
 assert points[6]['message_bytes']==17179869184 and points[6]['busbw_gbps']==315.00
 assert report['status']=='passed' and report['peak_bus_bw_gbps']==315.00
