"""Time-targeted hand interpolation; actuator commands are applied by SONIC's simulator."""
import numpy as np


class HandAgent:
    def __init__(self,events,preset,preparation=.9,settle=.12,release=None):
        self.events=sorted(events,key=lambda e:e['t_target'])
        self.preset=preset;self.preparation=preparation;self.settle=settle;self.release=release
        for prev,nxt in zip(self.events,self.events[1:]):
            if nxt['t_target']-preparation-settle < prev['t_target']:
                raise ValueError('hand target windows overlap')

    @staticmethod
    def weight(u):
        u=np.clip(u,0.,1.);return u**3*(10+u*(-15+6*u))

    def sample(self,t):
        previous=self.preset(0)
        for event in self.events:
            start=event['t_target']-self.preparation-self.settle
            if t<start:return previous
            target=self.preset(event['count'])
            if t<event['t_target']-self.settle:
                return previous+(target-previous)*self.weight((t-start)/self.preparation)
            previous=target
        if self.release is not None and t>=self.release:
            return previous+(self.preset(0)-previous)*self.weight((t-self.release)/self.preparation)
        return previous
