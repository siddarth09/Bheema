import numpy as np
from dataclasses import dataclass 

@dataclass
class WalkingGaitParams:
    """
    Parameters defining a walking gait 

    step_time:
        Duration of one full step (seconds)
    ds_fraction:
        Fraction of step_time spent in double support
    start_with_left_stance: 
        Single support phase has LEFT foot in stance
    """

    step_time:float 
    ds_fraction: float 
    start_with_left_stance: float 



class ContactSchedule: 
    """
    Walking Contact schedule for a biped humanoid
    Output:
        [Left-foot,Rightfoot] => [0/1,0/1]
    """

    def __init__(self,params:WalkingGaitParams):

        self.params = params
        self._validate()

        self.T = params.step_time 
        self.ds_frac = params.ds_fraction
        
        # Time partition for a single step 
        self.T_ds = self.ds_frac * self.T #Time for double support
        self.T_ss = self.T - self.T_ds 
        self.T_ds_pre = 0.5 * self.T_ds 
        self.T_ds_post = 0.5 * self.T_ds 


    def build(self,t0:float,dt:float,N:int):
        """
        Contact table over horizon 
        
        t0: current time 
        dt: MPC discretization timestep 
        N: MPC horizon length 

        return: 
            contact table: shape(2,N)
        """

        contact = np.zeros((2,N),dtype=np.int32)

        for k in range(N):
            t = t0+k*dt 
            cL,cR = self._contact_at_time(t)
            contact[0,k] = cL 
            contact[1,k] = cR 

        return contact 
    

    # ====================================
    # CORE
    # =====================================

    def _validate(self):
        if self.params.step_time<= 0.0:
            raise ValueError("step time is negative")
        if not(0.0<self.params.ds_fraction<1.0):
            raise ValueError("Fraction must be between (0,1)")
        

    def _contact_at_time(self,t:float):

        phase = self._phase_in_step(t)

        if phase["mode"] == "DS":
            return 1,1
        
        if phase["stance"] == "LEFT":
            return 1,0
        else:
            return 0,1
        

    def _phase_in_step(self,t:float):
        """
        Determining which step phase we are in 
        """

        step_idx = int(np.floor(t/self.T))

        tau = t - step_idx * self.T 

        if tau < self.T_ds_pre: 
            return {
                "step":step_idx,
                "mode": "DS",
                "stance": "BOTH"
            }
        

        if tau < self.T_ds_pre + self.T_ss:
            left_first = self.params.start_with_left_stance 
            left_stance = left_first if (step_idx % 2 == 0) else (not left_first)

            return{

                "step": step_idx,
                "mode": "SS",
                "stance": "LEFT" if left_stance else "RIGHT"
            }   
        
        return {
            "step": step_idx,
            "mode": "DS",
            "stance": "BOTH"
        }


def test_contact_schedule():
    
    params = WalkingGaitParams(
        step_time=0.6,
        ds_fraction=0.2,
        start_with_left_stance=True
    )

    sched = ContactSchedule(params)

    dt = 0.02
    N = 60
    t0 = 0.0

    C = sched.build(t0=t0, dt=dt, N=N)  # shape (2, N)

    print("Contact table shape:", C.shape)

    for k in range(N):
        cL, cR = C[:, k]
        total = cL + cR

        # 1) binary check
        assert cL in (0, 1) and cR in (0, 1)

        # 2) at least one contact
        assert total >= 1, f"Flight detected at k={k}"

        # 3) max two contacts
        assert total <= 2

    print("Contact table passed logical checks.")


if __name__ == "__main__":
    test_contact_schedule()
