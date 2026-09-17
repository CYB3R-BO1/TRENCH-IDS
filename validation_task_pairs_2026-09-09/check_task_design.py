import sys
sys.path.insert(0, 'src')
import json
import pandas as pd
from trench_ids import task_design as td

order = ['Scanning','DDoS','Reconnaissance','XSS','DoS','Password','Injection','Bot','BruteForce','Infiltration']
vals = [
[1.000,-0.291,0.628,0.148,-0.166,-0.363,-0.334,0.009,-0.532,0.401],
[-0.291,1.000,-0.154,-0.361,-0.183,0.021,-0.058,0.193,-0.005,-0.462],
[0.628,-0.154,1.000,-0.096,-0.057,-0.195,-0.383,-0.023,-0.451,0.197],
[0.148,-0.361,-0.096,1.000,0.235,-0.301,-0.028,-0.252,-0.390,0.707],
[-0.166,-0.183,-0.057,0.235,1.000,-0.137,-0.343,-0.415,0.022,0.161],
[-0.363,0.021,-0.195,-0.301,-0.137,1.000,0.468,-0.046,0.045,-0.457],
[-0.334,-0.058,-0.383,-0.028,-0.343,0.468,1.000,-0.066,0.062,-0.186],
[0.009,0.193,-0.023,-0.252,-0.415,-0.046,-0.066,1.000,-0.297,-0.075],
[-0.532,-0.005,-0.451,-0.390,0.022,0.045,0.062,-0.297,1.000,-0.516],
[0.401,-0.462,0.197,0.707,0.161,-0.457,-0.186,-0.075,-0.516,1.000],
]
sim = pd.DataFrame(vals, index=order, columns=order)
sizes = {'Scanning':3781419,'DDoS':3416504,'Reconnaissance':2620999,'XSS':2455020,'DoS':1196608,'Password':1153323,'Injection':684897,'Bot':143097,'BruteForce':120912,'Infiltration':116361}
g_nosize, s_nosize = td.assign_groups(sim, 0.35, None)
g_size, s_size = td.assign_groups(sim, 0.35, sizes)
locked = {1:['Scanning'],2:['Reconnaissance'],3:['DDoS','Infiltration'],4:['DoS','Injection'],5:['Password','Bot'],6:['XSS','BruteForce']}
def norm(gs):
    return sorted([sorted(g) for g in gs])
out = {'no_sizes': g_nosize, 'no_sizes_scores': s_nosize, 'with_sizes': g_size, 'with_sizes_scores': s_size, 'locked': locked, 'with_sizes_matches_locked': norm(g_size) == norm([locked[i] for i in [1,2,3,4,5,6]]), 'no_sizes_matches_locked': norm(g_nosize) == norm([locked[i] for i in [1,2,3,4,5,6]])}
print('no-sizes: ' + str(g_nosize))
print('with-sizes: ' + str(g_size))
print('with-sizes matches locked: ' + str(out['with_sizes_matches_locked']))
print('no-sizes matches locked: ' + str(out['no_sizes_matches_locked']))
with open('validation_task_pairs_2026-09-09/03_task_design_grouping.json','w') as f:
    json.dump(out, f, indent=2)
