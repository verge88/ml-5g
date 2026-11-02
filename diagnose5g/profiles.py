"""Evaluator-only bounded actuation, never imported by feature extraction."""
import random


def trials(replica):
    names=[(family,repeat) for family in ['healthy','user_path','application','combined'] for repeat in range(2)]
    random.Random(f'diagnostic-order:{replica}').shuffle(names)
    return names


def configuration(replica,family,repeat):
    r=random.Random(f'diagnostic:{replica}:{family}:{repeat}')
    total=r.uniform(110,145)
    return dict(family=family,repeat=repeat,start=r.randint(9,13),ramp=r.randint(5,8),
                path_peak=total/2 if family=='user_path' else total/4 if family=='combined' else 0.,
                app_peak=total if family=='application' else total/2 if family=='combined' else 0.)


def command(config,tick):
    level=max(0.,min(1.,(tick-config['start'])/config['ramp']))
    return dict(delay_ms=round(config['path_peak']*level,3),app_wait_ms=round(config['app_peak']*level,3),count=3,restart=False)
