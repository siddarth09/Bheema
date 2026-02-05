import casadi as ca 
import numpy as np 
import scipy.sparse as sp 
import time 
 

class CentroidalMPC:
    """
    Convex MPC for Bheema 

    decision variables:
        z = [x_1,x_2,...,x_N,u_0,.....u_{N-1}]

    State:
        x = [p_com,rpy,v_com,omega]
    
    Control Input:
        u = [f_l,f_r]
    """

    def __init__(self,N:int,dt:float):

        self.N = N
        self.dt = dt 
        self.nx = 12 
        self.nu = 6 

        self.nz = N*self.nx + N*self.nu 

        # Cost weights 
        # State tracking 
        self.Q = np.diag([50,50,200,
                          20,20,50,
                          5,5,10,
                          2,2,2])
        # Force regularization
        self.R = 1e-4* np.eye(self.nu)

        # Contact friction 

        self.mu = 0.6 
        self.fz_min = 10.0 
        self.fz_max = 1500.0

        self.opts = {
            "osqp":{
                "verbose":False,
                "eps_abs":1e-4,
                "eps_rel":1e-4,
                "max_iter":1000
            },
            "warm_start_primal":True,
            "warm_start_dual":True
        }


        self._static_cost()
        self._static_shift_matrix()
        self._static_friction_matrix()
        self._initialized = False 



    # Cost matrix 
    
    def _static_cost(self):
        """
        H belongs to R^(N*nx+*nu)x(N*nx+N*nu)
        H = Second derivative of the cost function 
        g = linear cost term

        Hessian matrix is just Q and R block places in correct global positions

        """

        rows,cols,vals = [],[],[]

        # State cost
        # Appending Q to H 
        for k in range(self.N):
            base = k * self.nx 
            for i in range(self.nx):
                if self.Q[i,i]!=0:
                    rows.append(base+i)
                    cols.append(base+i)
                    vals.append(2*self.Q[i,i])
        
        # Control Cost
        # Appending R to H
        for k in range(self.N):
            base = self.N * self.nx + k*self.nu 
            for i in range(self.nu):
                if self.R[i,i] != 0:
                    rows.append(base+i)
                    cols.append(base+i)
                    vals.append(2*self.R[i,i])

        # Builds the bigger diagnol matrix 
        self.H = ca.DM.triplet(
            rows,cols,ca.DM(vals),
            self.nz,self.nz
        )

        self.H_sp = self.H.sparsity()


    def _static_shift_matrix(self):
        """
        Builds a shift matrix such that
            x_k - A*x{k-1}-B*u_{k-1} = g
        """

        I = ca.DM.eye(self.N * self.nx)
        # Shifts the states one timestep back
        S = sp.kron(
            sp.diag([np.ones(self.N-1)],[-1]),
            sp.eye(self.nx)
        )

        self.S = ca.DM(
            ca.Sparsity(S.shape[0],S.shape[1],S.indptr,S.indices),
            S.data
        )

        self.I_block = I 

    
    def _static_friction_matrix(self):
        """
        Friction pyramid Constraints:

            |fx|< mu*fz 
            |fy|< mu*fz
        """

        rows,cols,vals = [],[],[]
        row = 0
        base_u = self.N * self.nx 


        for k in range(self.N):
            u0 = base_u + k * self.nu 

            for foot in range(2):
                fx = u0 + 3*foot 
                fy = fx + 1
                fz = fx + 2 


                # fx - mu*fz <= 0 

                rows += [row,row]
                cols += [fx,fz]
                vals += [1.0,-self.mu]
                row += 1

                # -fx - mu*fz<=0 

                rows += [row,row]
                cols += [fx,fz]
                vals += [-1.0,-self.mu]
                row += 1

                # fy - mu*fz <= 0 
                rows += [row,row]
                cols += [fy,fz]
                vals += [1.0,-self.mu]
                row += 1

                # -fy-mu*fz <=0 
                rows += [row,row]
                cols += [fy,fz]
                vals += [-1.0,-self.mu]
                row += 1


        A = sp.csc_matrix((vals,(rows,cols)),shape=(row,self.nz))

        self.A_friction = ca.DM(ca.Sparsity(A.shape[0],A.shape[1],
                                            A.indptr,A.indices),
                                A.data)
        


    def build_and_solve(self,Ad_seq,Bd_seq,gd_seq,x0,x_ref,contact_table):
        """MPC SOLVING

        MPC is solving:
        1. The sequence of forces over the horizon
        2. COM trajectory 
        3. Rotational stability 
        4. Force distribution
        """

        t0 = time.time()
        # Dynamic constraints 
        # Aeq*z = Beq
        Aeq, Beq = self._dynamic_constraints(
            Ad_seq,Bd_seq,gd_seq,x0 
        )


        # Cost gradient 
        # -2*Q*xref
        g = self._cost_gradient(x_ref) 

        # Bounds 
        # Foot contact awareness,which applies Box constraint 
        lbx,ubx = self._enforcing_bounds(contact_table)


        # Stacking the constraints 

        A = ca.vertcat(Aeq,self.A_friction)

        lb = ca.vertcat(Beq,-ca.inf*ca.DM.ones(self.A_friction.size(),1))
        ub = ca.vertcat(Beq,ca.DM.zeros(self.A_friction.size(),1))


        # Solver 

        if not self._initialized:
            qp = {"h": self.H_sp,"a":A.sparsity()}
            self.solver = ca.conic("mpc","osqp",qp,self.opts)
            self._initialized = True 

        # Solving the QP
        sol = self.solver(
            h = self.H,
            g = g,
            a = A,
            lba = lb,
            uba = ub,
            lbx = lbx,
            ubx = ubx

        )
        self.solve_time_ms = 1e3*(time.time()-t0)

        z = sol["x"].full().flatten()

        u0 = z[self.N*self.nx:self.N*self.nx+self.nu]

        return u0 
    


    # Utils 

    def _dynamic_constraints(self,Ad_seq,Bd_seq,gd_seq,x0):
        """
        x_k = A x_{k-1} + B u_{k-1} + g
        """

        rows = []
        rhs = []

        for k in range(self.N):
            row = ca.DM.zeros(self.nx,self.nz)

            # x_k 

            row[:,k*self.nx:(k+1)*self.nx] = ca.DM.eye(self.nx)
            if k == 0:
                rhs.append(Ad_seq[0]@ x0 + gd_seq[0])
            else:
                row[:,(k-1)*self.nx:k*self.nx] = -Ad_seq[k]
                rhs.append(gd_seq[k])

            #u_k 

            u_start = self.N * self.nx + k * self.nu 
            row[:,u_start:u_start+self.nu]=-Bd_seq[k]
            rows.append(row)

            Aeq = ca.vertcat(*rows)
            Beq = ca.vertcat(*rhs)

            return Aeq,Beq 
        

    def _cost_gradient(self,x_ref):
        
        g = ca.DM.zeros(self.nz,1)
        for k in range(self.N):
            idx = k*self.nx 
            g[idx:idx+self.nx] = -2 *ca.DM(self.Q)@ x_ref[:,k]

        return g 
    

    def _enforcing_bounds(self,contact):
        lbx = -ca.inf * ca.DM.ones(self.nz, 1)
        ubx =  ca.inf * ca.DM.ones(self.nz, 1)

        base_u = self.N * self.nx

        for k in range(self.N):
            for foot in range(2):
                fx = base_u + k*self.nu + 3*foot
                fy = fx + 1
                fz = fx + 2

                if contact[foot, k] == 0:
                    lbx[fx:fx+3] = 0
                    ubx[fx:fx+3] = 0
                else:
                    lbx[fz] = self.fz_min
                    ubx[fz] = self.fz_max

        return lbx, ubx