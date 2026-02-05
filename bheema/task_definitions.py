import numpy as np 
import pinocchio as pin 


class Task:
    """
   Base class for all whole body control tasks
    """
    def __init__(self,robot_model: pin.Model):
        self.model = robot_model 

        self.pelvis_frame = self.model.getFrameId("pelvis")
        self.left_foot_frame = self.model.getFrameId("left_foot")
        self.right_foot_frame = self.model.getFrameId("right_foot")


    def foot_position_task(self,
                           data: pin.Data,
                           frame_id: int, 
                           p_des: np.ndarray,
                           weight: float):
        """
        Regulate foot position in world frame

        Task: 
            p_foot(q) -> p_des
        """

        oMf = data.oMf[frame_id]
        p_cur = oMf.translation

        e = (p_des-p_cur).reshape(3,1)
        J6 = pin.computeFrameJacobian(
            self.model,data,frame_id,
            pin.ReferenceFrame.WORLD
        )
        J = J6[0:3,:]

        return {
            "J":J,
            "e":e,
            "W": weight,
            "name":"foot_position"
        }
    

    def pelvis_orientation_task(self,
                                data: pin.Data,
                                frame_id: int,
                                rpy_des: np.ndarray,
                                weight: float):
        
        """
        Keeping the torso upright and balancing the torso
        """
        oMf = data.oMf[self.pelvis_frame]
        R = oMf.rotation 
        rpy_cur = pin.rpy.matrixToRpy(R)
        e = (rpy_des[0:2]-rpy_cur[0:2]).reshape(2,1)

        J6 = pin.computeFrameJacobian(self.model,data,
                                      self.pelvis_frame,
                                      pin.ReferenceFrame.WORLD)
        
        J = J6[3:5,:]

        return{
            "J":J,
            "e":e,
            "w":weight,
            "name":"pelvis_orientation"
        }
    
    def posture_regularization(self,
                               q:np.ndarray,
                               q_nominal: np.ndarray,
                               weight: float):
        """
        Joint space posture regularization
        Prevents:
            - Knee collapse 
            - Keeps arms/Torso 
            
        """
        

        e = (q_nominal- q).reshape(-1,1)
        J = np.eye(self.model.nv)

        return {
            "J":J,
            "e":e,
            "w": weight,
            "name": "posture"
        }
    

