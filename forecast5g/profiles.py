"""Bounded actuator profiles, deliberately excluded from predictor inputs."""
import random

def configuration(replica,case,smoke=False):
    r=random.Random(f'{replica}:{case}:forecast-v1')
    return dict(case=case,ticks=36 if smoke else 72,start=r.randint(6,8) if smoke else r.randint(16,24),
        ramp=r.randint(8,10) if smoke else r.randint(20,28),peak=r.uniform(140,165),base=r.uniform(0,5),
        restart_tick=r.randint(3,5) if smoke else r.randint(8,13),phase=r.randint(0,20),power=r.choice([.8,1.,1.3]),
        transient_tick=12 if smoke else r.randint(22,30))

def command(c,t):
    name=c['case'];z=max(0,min(1,(t-c['start'])/c['ramp']))
    high=c['peak'] if name.startswith('gradual') else 64.
    if name.startswith('gradual') or name=='bounded_ramp':
        delay=c['base']+high*z**c['power']
        after=t-(c['start']+c['ramp']+7)
        if after>0:delay=c['base']+high*max(0,1-after/10)
    elif name=='transient_delay':delay=120. if t in [c['transient_tick'],c['transient_tick']+12] else c['base']
    else:delay=c['base']+2*((t+c['phase'])%3)
    count=6 if ('load' in name and ((t+c['phase'])//6)%2) else 3
    return dict(delay_ms=round(min(180,max(0,delay)),3),count=count,restart=('restart' in name and t==c['restart_tick']))

def case_order(replica,cases):
    values=list(cases);random.Random(replica+20261102).shuffle(values);return values
